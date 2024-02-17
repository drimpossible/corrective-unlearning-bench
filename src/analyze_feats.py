
def get_features(opt, prenet, loader, save_path):
    if not isfile(save_path+'_features.npy'):
        print('==> Generating features..')
        all_features = []

        for images, targets in loader:
            with torch.inference_mode():
                images, targets = images.cuda(), targets.cuda()
                features = prenet(images)
                all_features.append(features.detach().cpu().numpy())
            
        all_features = np.concatenate(all_features, axis=0)
        np.save(save_path+'_features.npy', all_features)
    else:
        print('==> Loading features..')
        all_features = np.load(save_path+'_features.npy')
    return all_features
