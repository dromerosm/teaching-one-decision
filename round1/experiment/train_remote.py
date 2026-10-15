"""Full fine-tune using Laya 0.4.2 primitives and explicit grouped holdouts."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import dataclasses,gc,hashlib,importlib.metadata,json,math,shutil,time,traceback
from pathlib import Path
import torch
ROOT=Path('/content/laya-finetune');DATA=ROOT/'data'
REPO='convaiinnovations/laya-multilingual';REV='1720e3e3357cfe1e281542e223f8273b0890ca34'
torch.set_num_threads(2);torch.manual_seed(42)
report={'status':'running','stage':'preflight','started_at':time.time(),'base_repo':REPO,'base_revision':REV,'epochs':[],
        'selection':'Highest validation all-fields accuracy; then decision accuracy; then earlier epoch. Test is never read by this script.',
        'versions':{p:importlib.metadata.version(p) for p in ['laya','torch','transformers']}}
def save():
    tmp=ROOT/'train_report.tmp';tmp.write_text(json.dumps(report,indent=2,default=str));tmp.replace(ROOT/'train_report.json')
def record_scores(records):
    correct=[int(int(z.argmax())==int(max(range(len(target)),key=lambda i:target[i]))) for qt,z,target,k in records]
    assert len(correct)%3==0
    return {'decision_accuracy':sum(correct)/len(correct),'all_fields_accuracy':sum(all(correct[i:i+3]) for i in range(0,len(correct),3))/(len(correct)//3),
            'per_field_accuracy':{key:sum(correct[j::3])/(len(correct)//3) for j,key in enumerate(['category','priority','needs_reply'])}}
save()
try:
    from huggingface_hub import snapshot_download
    from laya.train import TrainConfig,load_checkpoint,items_from_rows,read_data,train_model,save_checkpoint,calibration_records,calibration_report,evaluate_records,optimizer_updates,prior_collapse_message
    from laya.calibrate import fit_temperature_map
    assert 'T4' in torch.cuda.get_device_name(0)
    report['gpu']=torch.cuda.get_device_name(0)
    assignment=json.loads((DATA/'family_assignment.json').read_text())
    assert {s:sum(v==s for v in assignment.values()) for s in ['train','validation','calibration','test']}=={'train':50,'validation':10,'calibration':10,'test':10}
    report['data_sha256']={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['train.jsonl','validation.jsonl','calibration.jsonl','schema.json','questions.json','family_assignment.json']}
    report['stage']='download';save();t=time.perf_counter()
    path=snapshot_download(REPO,revision=REV,allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    report['download_seconds']=time.perf_counter()-t
    report['stage']='load_and_preprocess';save();t=time.perf_counter()
    model,tok,cfg=load_checkpoint(path);device=torch.device('cuda');model.to(device)
    config=TrainConfig(epochs=4,micro_batch=8,grad_accum=4,loss='soft-ce',freeze_encoder=False,
                       encoder_lr=2.5e-5,head_lr=1e-4,seed=42,max_len=1024,head_max_len=256,
                       calib_max=0,calib_frac=0.0,amp=True,gradient_checkpointing=True,log_every=25)
    config.validate();report['train_config']=dataclasses.asdict(config)
    items={};report['item_counts']={};report['skipped']={}
    for split,n in [('train',2000),('validation',400),('calibration',400)]:
        rows=read_data(str(DATA/f'{split}.jsonl'));assert len(rows)==n
        items[split],skipped=items_from_rows(tok,rows,1024,256)
        assert len(items[split])==n*3 and not skipped,(split,len(items[split]),skipped)
        report['item_counts'][split]=len(items[split]);report['skipped'][split]=skipped
    report['optimizer_updates_planned']=optimizer_updates(len(items['train']),config.micro_batch,config.grad_accum,config.epochs)
    report['load_preprocess_seconds']=time.perf_counter()-t
    report['stage']='baseline_validation';save();t=time.perf_counter()
    baseline=calibration_records(model,tok,items['validation'],device,1024,256,batch_size=8)
    report['baseline_validation']=record_scores(baseline)
    report['baseline_validation_seconds']=time.perf_counter()-t
    best=(-1.0,-1.0);epoch_start=time.perf_counter()
    def epoch_done(epoch,loss):
        global best,epoch_start
        torch.cuda.synchronize();trained_seconds=time.perf_counter()-epoch_start
        assert math.isfinite(loss),loss
        start=time.perf_counter()
        records=calibration_records(model,tok,items['validation'],device,1024,256,batch_size=8)
        scores=record_scores(records);key=(scores['all_fields_accuracy'],scores['decision_accuracy'])
        row={'epoch':epoch+1,'mean_loss':loss,'training_seconds':trained_seconds,'validation':scores,'validation_seconds':time.perf_counter()-start}
        if key>best:
            best=key
            save_checkpoint(model,tok,dict(cfg,max_len=1024,head_max_len=256),str(ROOT/'checkpoint'))
            report['selected_epoch']=epoch+1;report['selected_validation']=scores
        report['epochs'].append(row);save()
        print('EPOCH_RESULT '+json.dumps(row),flush=True)
        model.train()  # calibration_records switches to eval; resume training with dropout.
        epoch_start=time.perf_counter()
    report['stage']='training';save();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    report['epoch_loss']=train_model(model,tok,items['train'],config,device,1024,256,on_epoch_end=epoch_done)
    torch.cuda.synchronize();report['training_with_validation_seconds']=time.perf_counter()-start
    report['peak_training_gpu_bytes']=torch.cuda.max_memory_allocated()
    del model;gc.collect();torch.cuda.empty_cache()
    report['stage']='calibration';save();start=time.perf_counter()
    model,tok,selected_cfg=load_checkpoint(str(ROOT/'checkpoint'));model.to(device)
    calib=calibration_records(model,tok,items['calibration'],device,1024,256,batch_size=8)
    fitted=fit_temperature_map(calib)
    report['calibration']=calibration_report(calib,fitted['temperature'])
    report['calibration_collapse_warning']=prior_collapse_message(calib)
    report['calibration_fit_metrics']=evaluate_records(calib,fitted['temperature'],fitted['temperature_by_options'])
    selected_cfg.update(fine_tuned=True,temperature=fitted['temperature'])
    for key in ['temperature_by_options','binning_map']:selected_cfg.pop(key,None)
    if fitted['temperature_by_options']:selected_cfg['temperature_by_options']=fitted['temperature_by_options']
    selected_cfg['training']={'recipe':dataclasses.asdict(config),'selected_epoch':report['selected_epoch'],
                              'train_rows':2000,'validation_rows':400,'calibration_rows':400,'test_rows_used':0,'data_sha256':report['data_sha256'],
                              'calibration':report['calibration']}
    # Temperatures change configuration only; preserve the already selected weights.
    (ROOT/'checkpoint'/'rl_agent_config.json').write_text(json.dumps(selected_cfg,indent=2))
    shutil.copy2(DATA/'questions.json',ROOT/'checkpoint'/'questions.json')
    shutil.copy2(DATA/'schema.json',ROOT/'checkpoint'/'schema.json')
    report['calibration_seconds']=time.perf_counter()-start
    report['checkpoint_sha256']=hashlib.sha256((ROOT/'checkpoint'/'model.safetensors').read_bytes()).hexdigest()
    report['status']='complete';report['stage']='complete';report['finished_at']=time.time();save()
    shutil.copy2(ROOT/'train_report.json',ROOT/'checkpoint'/'train_report.json')
    print('TRAINING_COMPLETE '+json.dumps({k:report[k] for k in ['selected_epoch','selected_validation','checkpoint_sha256']}),flush=True)
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();report['finished_at']=time.time();save();raise
