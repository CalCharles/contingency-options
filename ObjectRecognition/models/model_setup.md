# Model Configuration by Name

**Note** some of old models are not tested. the info might be inaccurate.

## atari breakout

42531_2.npy
- two_layer_5_5_old.json
- binarize=0.1
- with paddle_bin.npy + remove mean

paddle_bin.npy
- two_layer
- binarize=0.1

paddle_bin_smooth.pth
- attn_softmax.json
- binarize=0.1

42531_2_smooth.pth
- attn_softmax.json
- binarize=0.1
- with paddle_bin_smooth.npy + remove mean

42531_2_smooth_2.pth
- attn_softmax.json
- binarize=0.1
- with paddle_bin_smooth.npy + remove mean

## self breakout

The followings have similar setup
```
ball_bin_long.npy
ball_bin.npy
ball.npy
paddle_bin.npy
paddle.npy
ball_bin_long_smooth.pth
ball_bin_smooth.npy
paddle_bin_long_smooth_2.pth
paddle_bin_old.npy
```
- two_layer.json if `.npy` and attn_base.json if `.pth` (or `_smooth`)
- binarize=0.1 if `_bin`
- train with complicated blocks if `_long`
- ball models should be attached to a remove mean paddle