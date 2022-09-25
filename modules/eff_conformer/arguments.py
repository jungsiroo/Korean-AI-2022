import argparse
import torch

def get_args():
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=20)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--save_result_every', type=int, default=10)
    args.add_argument('--checkpoint_every', type=int, default=1)
    args.add_argument('--print_every', type=int, default=50)
    args.add_argument('--dataset', type=str, default='kspon')
    args.add_argument('--output_unit', type=str, default='character')
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--num_threads', type=int, default=16)
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--max_grad_norm', type=int, default=400)
    args.add_argument('--warmup_steps', type=int, default=1000)
    args.add_argument('--weight_decay', type=float, default=1e-05)
    args.add_argument('--reduction', type=str, default='mean')
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--total_steps', type=int, default=200000)

    args.add_argument('--architecture', type=str, default='deepspeech2')
    args.add_argument('--use_bidirectional', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--joint_ctc_attention', type=bool, default=False)

    args.add_argument('--audio_extension', type=str, default='pcm')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    args.add_argument('--pre_mode', type=str, default="phonetic")

    ### For Efficient Conformer
    args.add_argument("--rank", type=int, default=0)

    args.add_argument("-c", "--config_file",type=str,
    default="modules/eff_conformer/configs/EfficientConformerCTCSmall.json",
    help="Json configuration file containing model hyperparameters")

    args.add_argument("-i", "--initial_epoch",type=int,default=0,
    help="Load model from checkpoint")

    args.add_argument("--initial_epoch_lm",type=str,default=None,
    help="Load language model from checkpoint")

    args.add_argument("--initial_epoch_encoder",type=str,default=None,
    help="Load model encoder from encoder checkpoint")

    args.add_argument("-p", "--prepare_dataset",action="store_true",
    help="Prepare dataset for training")
    args.add_argument("-d", "--distributed", action="store_true",)
    args.add_argument("--world_size",type=int,default=torch.cuda.device_count(),help="Number of available GPUs")
    args.add_argument("--create_tokenizer",action="store_false",help="Create model tokenizer")
    args.add_argument("--verbose_val",action="store_true",help="Evaluation verbose")
    args.add_argument("--val_steps",type=int,default=None,help="Number of validation steps")
    args.add_argument("--show_dict",action="store_true",help="Show model dict summary")
    args.add_argument("--swa",action="store_true",help="Stochastic weight averaging")
    args.add_argument("--swa_epochs",nargs="+", default=None, help="Start epoch / end epoch for swa")
    args.add_argument("--swa_epochs_list", nargs="+", default=None, help="List of checkpoints epochs for swa")
    args.add_argument("--swa_type",type=str, default="equal", 
    help="Stochastic weight averaging type (equal/exp)")
    args.add_argument("--parallel",action="store_false",help="Parallelize model using data parallelization")
    args.add_argument("--rnnt_max_consec_dec_steps", type=int,default=None,
    help="Number of maximum consecutive transducer decoder steps during inference")
    args.add_argument("--eval_loss", action="store_true", 
    help="Compute evaluation loss during evaluation")
    args.add_argument("--gready", action="store_false", help="Proceed to a gready search evaluation")
    args.add_argument("--saving_period", type=int, default=1, help="Model saving every 'n' epochs")
    args.add_argument("--val_period", type=int, default=1, help="Model validation every 'n' epochs")
    args.add_argument("--profiler", action="store_true",help="Enable eval time profiler")
    args.add_argument("--batch_size_eval",type=int,default=8, help="Evaluation batch size")

    config = args.parse_args()
    return config