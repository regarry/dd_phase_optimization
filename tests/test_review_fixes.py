import random
import numpy as np
import pytest
import torch
from PIL import Image
from physics.phase_io import quantize_phase, save_codes, load_phase, resize_phase
from physics.phase_mask_norm import process_phase_mask, normalize_phase_image
from training.training import (BinaryDiceLoss, MultiClassDiceLoss, compute_total_loss,
                               DynamicSpatialWellLoss, EarlyStopping)
from training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint, resume_config
from models.wrappers import ParallelEndToEndModel

@pytest.mark.parametrize('classes', [1, 3])
@pytest.mark.parametrize('spatial', [(4,5), (2,4,5)])
def test_dice_dispatch_and_gradient(classes, spatial):
    logits = torch.randn((2,classes,*spatial), requires_grad=True)
    target = torch.randint(classes if classes>1 else 2, (2,1,*spatial))
    criterion = BinaryDiceLoss() if classes == 1 else MultiClassDiceLoss()
    loss, _ = compute_total_loss(logits,target,None,{'dice_loss_weight':1}, {'dice_loss':criterion})
    reference = criterion(logits,target)
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(torch.autograd.grad(loss,logits,retain_graph=True)[0],
                               torch.autograd.grad(reference,logits)[0])

@pytest.mark.parametrize('bits', [8,10])
def test_export_training_agree(tmp_path,bits):
    phase = np.array([[-.1,0,.3,np.pi,2*np.pi-.001]],dtype=np.float64)
    p=tmp_path/'codes.tif'
    codes=save_codes(phase,p,bits)
    assert codes.dtype == (np.uint8 if bits==8 else np.uint16)
    loaded=load_phase(p)
    expected=quantize_phase(torch.tensor(phase),bits).numpy()
    np.testing.assert_allclose(loaded,expected,atol=3e-7)
    assert codes.max()==(1<<bits)-1
    x=torch.tensor(phase,requires_grad=True)
    quantize_phase(x,bits).sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x))

@pytest.mark.parametrize('bits', [8,10])
def test_export_entry_points(tmp_path,bits):
    phase=np.array([[0,.1,np.pi,4]],np.float32)
    source=tmp_path/'raw.tif';Image.fromarray(phase).save(source)
    a,b=tmp_path/'array.png',tmp_path/'file.png'
    process_phase_mask(phase,a,bit_depth=bits)
    normalize_phase_image(source,b,bit_depth=bits)
    np.testing.assert_array_equal(np.array(Image.open(a)),np.array(Image.open(b)))
    np.testing.assert_allclose(load_phase(a),load_phase(b))

def test_units_and_invalid_output(tmp_path):
    p=tmp_path/'legacy.tif';Image.fromarray(np.array([[0,128,200]],np.uint8)).save(p)
    with pytest.raises(ValueError,match='Legacy integer'):load_phase(p)
    np.testing.assert_allclose(load_phase(p,'drive_codes',8),np.array([[0,128,200]])/255*2*np.pi,atol=3e-7)
    with pytest.raises(ValueError,match='BMP'):save_codes(np.zeros((2,2)),tmp_path/'out.bmp',10)
    with pytest.raises(ValueError,match='integer'):quantize_phase(torch.ones(1),0)

def test_circular_resize_and_final_carrier(tmp_path):
    phase=np.array([[.01,2*np.pi-.01],[.01,2*np.pi-.01]])
    resized=resize_phase(phase,(9,2))
    assert np.abs(np.angle(np.exp(1j*resized))).max()<.02
    path=tmp_path/'carrier.tif'
    process_phase_mask(np.zeros((2,2)),path,8,2,False,True,4,0,0,10)
    codes=np.array(Image.open(path))
    np.testing.assert_array_equal(codes[0,:4],codes[0,4:])

def test_steepness_fixed_and_effective():
    x=torch.rand(2,3,10,requires_grad=True)
    a=DynamicSpatialWellLoss(1,3,.5);b=DynamicSpatialWellLoss(1,3,4)
    assert not list(a.parameters()) and isinstance(a.steepness,float)
    a(x).backward();assert x.grad is not None
    b(x);assert not torch.allclose(a.last_cost_map,b.last_cost_map)
    with pytest.raises(ValueError):DynamicSpatialWellLoss(1,3,0)

def test_eval_mode_preserved():
    model=ParallelEndToEndModel.__new__(ParallelEndToEndModel);torch.nn.Module.__init__(model)
    model.config={}
    class Physics(torch.nn.Module):
        def forward(self, phase, emitters):return emitters
    model.physics=Physics()
    model.unet=torch.nn.Sequential(torch.nn.BatchNorm2d(1),torch.nn.Dropout(.5))
    model.eval();before=model.unet[0].running_mean.clone()
    x=torch.randn(2,1,4,4)
    torch.testing.assert_close(model(None,x),model(None,x))
    assert not model.unet.training
    torch.testing.assert_close(before,model.unet[0].running_mean)
    model.train();model(None,x);assert model.unet.training

def components():
    model=torch.nn.Linear(2,1)
    phase=torch.nn.Parameter(torch.randn(2))
    optimizer=torch.optim.Adam([*model.parameters(),phase],lr=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=1)
    return model,phase,optimizer,scheduler,EarlyStopping(patience=10)

def step(parts):
    model,phase,opt,sched,stop=parts
    opt.zero_grad()
    x=torch.randn(3,2)*np.random.rand()+random.random()
    loss=(model(x+phase)**2).mean()
    loss.backward();opt.step();sched.step(loss.item());stop(loss.item())
    return loss.detach().clone()

def test_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(12);np.random.seed(12);random.seed(12)
    original=components();step(original)
    path=tmp_path/'latest.pt'
    save_checkpoint(path,*original,0,{'slm_bit_depth':10},{},[1],[2],[(torch.ones(1),torch.zeros(1))])
    expected=step(original)
    state=read_checkpoint(path);resumed=components()
    assert restore_checkpoint(state,*resumed)==1
    torch.testing.assert_close(step(resumed),expected,rtol=0,atol=0)
    for a,b in zip(original[0].parameters(),resumed[0].parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(original[1],resumed[1],rtol=0,atol=0)
    assert original[3].state_dict()==resumed[3].state_dict()
    assert vars(original[4])==vars(resumed[4])
    assert resume_config({'max_epochs':20},state)['slm_bit_depth']==10
    with pytest.raises(ValueError,match='slm_bit_depth'):resume_config({'slm_bit_depth':8},state)

def test_training_main_resume_integration(tmp_path,monkeypatch):
    import sys, yaml, copy
    import training.training as t
    class TinyModel(torch.nn.Module):
        def __init__(self,config):
            super().__init__();self.layer=torch.nn.Conv2d(1,1,1)
        def forward(self,phase,xyz):return self.layer(xyz+phase.mean())
    class TinyData(torch.utils.data.Dataset):
        def __init__(self,epoch_length,config):self.n=epoch_length
        def __len__(self):return self.n
        def __getitem__(self,i):
            x=torch.rand(1,2,2)+np.random.rand()+random.random()
            return x,(x>1.5).float()
    class NoSnapshot:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def dump(self):pass
        def end(self):pass
    original_loader=t.DataLoader
    monkeypatch.setattr(t,'DataLoader',lambda *a,**kw:original_loader(*a,**{**kw,'num_workers':0}))
    monkeypatch.setattr(t,'ParallelEndToEndModel',TinyModel)
    monkeypatch.setattr(t,'SyntheticMicroscopeData',TinyData)
    monkeypatch.setattr(t,'generate_bead_templates',lambda *a:None)
    monkeypatch.setattr(t,'get_initial_phase_mask',lambda c:np.zeros((2,2),np.float32))
    monkeypatch.setattr(t,'MemorySnapshot',NoSnapshot)
    def expand(c,path):c.update(training_results_dir=path,bead_volume=[2,2,2]);return c
    monkeypatch.setattr(t,'expand_config',expand)
    base=yaml.safe_load(open('config.yaml'))
    base.update(cnn_device='cpu',ce_loss_weight=1,ntrain=4,nvalid=4,
                accumulation_steps=2,slm_bit_depth=10,max_epochs=2)
    def run(config,name):
        path=tmp_path/f'{name}.yaml';path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(sys,'argv',['training.py','-c',str(path)])
        t.main()
    config={**base,'training_data_path':str(tmp_path/'split')}
    run(config,'first')
    checkpoint_path=next((tmp_path/'split').glob('*/checkpoints/latest.pt'))
    old=read_checkpoint(checkpoint_path)
    assert 'training_results_dir' not in old['raw_config']
    run({'resume_checkpoint':str(checkpoint_path),'max_epochs':3},'resume')
    actual=read_checkpoint(checkpoint_path)
    run({**base,'max_epochs':3,'training_data_path':str(tmp_path/'whole')},'whole')
    expected=read_checkpoint(next((tmp_path/'whole').glob('*/checkpoints/latest.pt')))
    assert actual['epoch']==2
    assert actual['train_losses']==expected['train_losses']
    assert actual['val_losses']==expected['val_losses']
    torch.testing.assert_close(actual['raw_phase'],expected['raw_phase'],rtol=0,atol=0)
    for key in actual['model']:torch.testing.assert_close(actual['model'][key],expected['model'][key],rtol=0,atol=0)

def test_rotation_preserves_10bit_contract(tmp_path):
    from RotateImage import process_phase_mask_and_save, rotate_bmp_image_and_rename
    source=tmp_path/'phase.tif'
    phase=np.array([[0,.2,1],[2,3,4],[5,5.5,6]],np.float32)
    Image.fromarray(phase).save(source)
    out=process_phase_mask_and_save(source,tmp_path,'rot',bit_depth=10)
    rotated=rotate_bmp_image_and_rename(out,90)
    np.testing.assert_allclose(np.exp(1j*load_phase(rotated)),np.rot90(np.exp(1j*load_phase(out))),atol=1e-6)

@pytest.mark.parametrize('config', [{}, {'gradient_accumulation_steps':32},
    {'accumulation_steps':2,'gradient_accumulation_steps':2},
    {'accumulation_steps':0}, {'accumulation_steps':-1}, {'accumulation_steps':True},
    {'accumulation_steps':1.5}, {'accumulation_steps':'2'}])
def test_accumulation_configuration_rejected(config):
    from training.training import require_accumulation_steps
    with pytest.raises(ValueError):require_accumulation_steps(config)

@pytest.mark.parametrize('optimizer_type', [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize('sample_count,batch_size,steps', [(8,2,2),(5,2,3),(7,2,3),(5,2,1)])
def test_accumulated_update_matches_combined_batch(optimizer_type,sample_count,batch_size,steps):
    import copy
    from training.training import train_one_epoch
    torch.manual_seed(51)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Conv2d(1,1,1)
        def forward(self,phase,x):return self.layer(x)+phase.mean()
    model=Tiny();reference=copy.deepcopy(model)
    phase=torch.nn.Parameter(torch.full((2,2),.3))
    reference_phase=torch.nn.Parameter(phase.detach().clone())
    optimizer=optimizer_type([*model.parameters(),phase],lr=.03)
    reference_optimizer=optimizer_type([*reference.parameters(),reference_phase],lr=.03)
    x=torch.randn(sample_count,1,2,2);y=torch.randint(2,x.shape).float()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=batch_size)
    config={'accumulation_steps':steps,'cnn_device':'cpu','slm_bit_depth':8,'ce_loss_weight':1}
    # Dirty gradients must be cleared before beginning the epoch.
    for p in [*model.parameters(),phase]:p.grad=torch.ones_like(p)*123
    reported=train_one_epoch(model,loader,optimizer,{'ce_loss':torch.nn.BCEWithLogitsLoss()},phase,config,0)
    weighted_loss=0
    for start in range(0,sample_count,batch_size*steps):
        end=min(start+batch_size*steps,sample_count)
        reference_optimizer.zero_grad()
        prediction=reference(quantize_phase(reference_phase,8),x[start:end])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(prediction,y[start:end])
        loss.backward();reference_optimizer.step()
        weighted_loss+=loss.item()*(end-start)
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(phase,reference_phase,rtol=1e-5,atol=1e-7)
    assert reported==pytest.approx(weighted_loss/sample_count,rel=1e-6)


def test_old_summed_gradient_checkpoint_rejected(tmp_path):
    path=tmp_path/'old.pt';torch.save({'format_version':1},path)
    with pytest.raises(ValueError,match='averaged-gradient'):read_checkpoint(path)
