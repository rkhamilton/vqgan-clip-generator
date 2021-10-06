# This contains the original math to generate an image from VQGAN+CLIP. I don't fully understand what it's doing and don't expect to change it.

def train(i):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()
    
    if i % args.display_freq == 0:
        checkin(i, lossAll)
       
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    
    #with torch.no_grad():
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))