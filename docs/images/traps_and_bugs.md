## Known traps (and where they are handled)

- **Sky floaters.** Without a sky model, Gaussians float in the sky. A cubemap is composited behind the splats (`model/sky.py`) and supervised with the sky mask.
- **Gaussian explosion / out of memory.** The Gaussian count is capped with gsplat `MCMCStrategy(cap_max)`, together with opacity and scale regularizers.
- **MCMC collapse on long drives.** MCMC jitters Gaussian positions with noise of size `lr_means × extent × noise_lr × covariance`, in metres. That grows with the square of the scene size. The opacity regularizer also pushes Gaussians toward low opacity, which is where the noise applies. Together, large budgets (≥3M) on a 78 m scene blow up: scales grow, opacities die, and `torch.multinomial` crashes. **Fix:** lower `train.noise_lr` (210 for the full AV2 log, vs gsplat's 5e5).
- **Exposure differences.** Cameras disagree on brightness. A per-image affine colour correction is applied (`model/exposure.py`); keep it close to identity.
- **Box jitter.** Tracked boxes wobble. An se(3) correction per track and frame is learned, with a smoothness prior.
- **LiDAR sees over cars.** The LiDAR sits above the cameras, so far returns leak through gaps in sparse depth and wreck the depth loss on cars. An occlusion filter in `data/depth.py` handles this.
- **Slow pose gradients.** `node_poses[node_ids]` has an atomics-bound backward pass. Use `gather_per_gaussian`.
- **Non-rigid actors.** Pedestrians and cyclists are masked out of the loss instead of modelled.
- **Lane markings.** Cityscapes has no lane-marking class. Use the Mapillary Vistas model (the default) or Waymo panoptic labels (`semantics_gt/`).
