
import numpy as np
import time
import pdb

import torch
import torch.nn as nn

from dataloader import *
from model import *
from model_generator import Generator
from options import *
from config import *
import utils


if __name__ == "__main__":
	args = parse_args()
	if args.debug:
		pdb.set_trace()

	config = Config(args)
	worker_init_fn = None

	if config.seed >= 0:
		utils.set_seed(config.seed)
		worker_init_fn = np.random.seed(config.seed)

	utils.save_config(config, os.path.join(config.output_path, "config.txt"))

	for split_no in range(config.num_splits):
		start_time = time.time()
		net = Model(config.num_classes, hidden_dim=128).cuda()
		generator = Generator(input_dim=100, output_dim=128, hidden_dim=128, num_classes=40)

		net = net.cuda()
		generator = generator.cuda()

		ckpt_path_source = os.path.join(config.ckpt_path_source, 'best_model_split_{}.pkl'.format(split_no))
		
		net.load_state_dict(torch.load(ckpt_path_source))
		net.requires_grad_(False)

		criterion = nn.CrossEntropyLoss().cuda()
		optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr[0])
		print(net)

		prev_epoch = 0
		prev_loss = 9999

		num_samples = config.batch_size_each * 8

		for n in range(1, config.num_epochs + 1):

			if n > 1 and config.lr[n - 1] != config.lr[n - 2]:
				print("[LR decayed: {} => {}]".format(config.lr[n - 2], config.lr[n - 1]))
				for param_group in optimizer.param_groups:
					param_group["lr"] = config.lr[n - 1]

			generator.train()
			optimizer.zero_grad()

			z = torch.randn((num_samples, 100)).cuda()

			labels = torch.randint(0, config.num_classes, (num_samples,)).cuda()

			z = z.contiguous()
			labels = labels.contiguous()

			fake_samples = generator(z, labels)

			output = net.classifier(fake_samples)

			loss = criterion(output, labels)

			loss.backward()
			optimizer.step()

			print('{:04d} epoch train loss: {:.4f}'.format(n, float(loss)))

			if loss < prev_loss and n // 10:
				prev_loss = loss
				prev_epoch = n
				print('> Minimum Loss {:.4f}'.format(prev_loss))

				torch.save(generator.state_dict(), os.path.join(config.model_path, "best_model_split_{}.pkl".format(split_no)))
				utils.save_best_record_loss(os.path.join(config.output_path, "best_score_split_{}.txt".format(split_no)), prev_epoch, loss)
				
		end_time = time.time()
		elapsed = end_time - start_time

		print('Split no. {} finished'.format(split_no))
		
		print('[Best] Minumum Loss {:.4f}'.format(prev_loss))
		print('Total elapsed time: {} hours {} mins'.format(int(elapsed) // 3600, (int(elapsed) % 3600) // 60))
