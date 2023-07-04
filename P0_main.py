import os
from config import get_config




# def main(args, save_dir):

#     # mode: [generation / extraction / selection / classification]
#     if args.mode == "":
#         net = A(args, save_dir)
#         net.train(args.mode)

#     elif args.mode == "":
#         net = B(args)
#         net.run()

#     elif args.mode == "":
#         net = C(args)
#         net.run()

#     elif args.mode == "":
#         net = D(args)
#         net.run()


#     else:
#         print('>> Please execute with mode: ex) --mode "selection"')
#         print('>> MODE: [generation / extraction / selection / classification]')


if __name__ == "__main__":

    config, unparsed = get_config()
    # run_time = date.today().strftime("%m-%d") + datetime.now().strftime("-%H-%M")
    # save_dir = config.save_path + run_time + "/"

    # store configuration
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(save_dir +  "config.txt", "w") as f:
    #     f.write("Parameters for " + config.mode + ":\n\n")
    #     for arg in vars(config):
    #         argname = arg
    #         contents = str(getattr(config, arg))
    #         # print(argname + ' = ' + contents)
    #         f.write(argname + " = " + contents + "\n")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    print(config)
    # main(config, save_dir)