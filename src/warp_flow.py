import torch 

def warp_flow(inputs, flows, device):

    flows = flows.permute(0,3,1,2)
    B, C, H, W = inputs.shape

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)

    yy = torch.arange(0, H).view(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)

    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx,yy),1).float().to(device)

    vgrid = grid + flows

    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0

    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    warped_outputs = torch.nn.functional.grid_sample(inputs, vgrid.permute(0,2,3,1), 'nearest')
    # import matplotlib.pyplot as plt



    # for input, warped_output in zip(inputs_, warped_outputs):
    #     fig , (ax0, ax1) = plt.subplots(1,2)
    #     ax0.imshow(torch.sigmoid(input).cpu().detach().permute(1,2,0),cmap='gray',vmin=0, vmax=1)
    #     ax1.imshow(torch.sigmoid(warped_output).cpu().detach().permute(1,2,0), cmap='gray',vmin=0, vmax=1)
    #     plt.show()
    return warped_outputs