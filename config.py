import argparse
from easydict import EasyDict as edict

def parse_configs():
    parser = argparse.ArgumentParser(description = 'Google-Landmark')
    parser.add_argument('--seed', type = int, default = 1138, 
                        help = 're-produce the results with seed random')
    
    ####################################################################
    #################     Model Config   ###############################
    ####################################################################
    parser.add_argument('--save_weights_only', type = bool, default = True, 
                        help = '')
    parser.add_argument('--backbone', default = 'gluon_seresnext101_32x4d',
                        help = 'Main pretrained-model')
    parser.add_argument('--embedding_size', type = int, default = 512,
                        help = 'Size of vector before full connected layer')
    parser.add_argument('--pool', default = 'gem',
                        help = 'Pooling type after backbone block')
    parser.add_argument('--neck', default = 'option-D',
                        help = 'Configuration of neck block, convert embedding vector -> n_classes')
    parser.add_argument('--n_classes', type = int, default = 1000, 
                        help = 'Define the number of classes')
    parser.add_argument('--head', default = 'arc_margin',
                        help = 'calculate cosine value')

    ###################################################################
    ################### Loss ##########################################
    ###################################################################
    parser.add_argument('--crit', default = 'focal', 
                        help = 'Loss for solving imbalanced data problem')
    parser.add_argument('--loss', default = 'arcface',
                        help = 'main loss for training')
    parser.add_argument('--argface.s', type = float, default = 30, 
                        help = 'config for arcface loss')
    parser.add_argument('--argface.m', type = float, default = 0.35,
                        help = 'config for arcface loss')

    ##################################################################
    ######################## Training ################################
    ##################################################################

    parser.add_argument('--num_epochs', type = int, default = 10,
                        help = 'the number of epoch')
    parser.add_argument('--batch_size', type = int, default = 32,
                        help = 'the number of samples for each each mini-batch')
    parser.add_argument('--path2weights', default = './models/weights.pt',
                        help = 'saving weight path')
    parser.add_argument('--p_trainable', type = bool, default = True)
    parser.add_argument('--pretrained_weights', default = None)


    config = edict(vars(parser.parse_args()))

    return config

if __name__ == '__main__':
    config = parse_configs()
    print(config)



