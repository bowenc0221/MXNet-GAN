import mxnet as mx

def get_symbol_generator():
    no_bias = True
    fix_gamma = True
    epsilon = 1e-5 + 1e-12

    rand = mx.sym.Variable('rand')

    g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4, 4), num_filter=1024, no_bias=no_bias)
    gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=epsilon)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=512,
                              no_bias=no_bias)
    gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=epsilon)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=256,
                              no_bias=no_bias)
    gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=epsilon)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=128,
                              no_bias=no_bias)
    gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=epsilon)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=3, no_bias=no_bias)
    generatorSymbol = mx.sym.Activation(g5, name='gact5', act_type='tanh')

    return generatorSymbol

def get_symbol_discriminator():
    no_bias = True
    fix_gamma = True
    epsilon = 1e-5 + 1e-12

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=128, no_bias=no_bias)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=256, no_bias=no_bias)
    dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=epsilon)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=512, no_bias=no_bias)
    dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=epsilon)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=1024,
                            no_bias=no_bias)
    dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=epsilon)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4, 4), num_filter=1, no_bias=no_bias)
    d5 = mx.sym.Flatten(d5)

    discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')

    return discriminatorSymbol