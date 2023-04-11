
import numpy as np
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from dataloader import *
from model import *
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

		trainset = [None] * (config.num_subjects - 1)

		i = 0
		for subject_id in range(config.num_subjects):
			if subject_id == config.target_subject:
				continue

			trainset_data = eegloader(os.path.join(config.data_path, config.data_file), os.path.join(config.data_path, config.split_file), dtype='train', split_no=split_no, seed=config.seed, subject_id=subject_id, target_subject=config.target_subject)
			
			batch_size = config.batch_size_each

			trainset[i] = DataLoader(trainset_data, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)
			i += 1

		i = 0
		valset_data = []
		testset_data = []
		for subject_id in range(config.num_subjects):
			if subject_id == config.target_subject:
				continue

			valset_data_each = eegloader(os.path.join(config.data_path, config.data_file), os.path.join(config.data_path, config.split_file), dtype='val', split_no=split_no, seed=config.seed, subject_id=subject_id, target_subject=subject_id)
			testset_data_each = eegloader(os.path.join(config.data_path, config.data_file), os.path.join(config.data_path, config.split_file), dtype='test', split_no=split_no, seed=config.seed, subject_id=subject_id, target_subject=subject_id)

			valset_data.append(valset_data_each)
			testset_data.append(testset_data_each)
			i += 1

		valset_data = ConcatDataset(valset_data)
		testset_data = ConcatDataset(testset_data)

		print(len(valset_data))
		print(len(testset_data))

		valset = DataLoader(valset_data, batch_size=config.batch_size_each, worker_init_fn=worker_init_fn, num_workers=4)
		testset = DataLoader(testset_data, batch_size=config.batch_size_each, worker_init_fn=worker_init_fn, num_workers=4)

		net = Model(config.num_classes, hidden_dim=128).cuda()
		criterion = nn.CrossEntropyLoss().cuda()
		optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0])
		print(net)

		prev_epoch = 0
		prev_loss = 9999
		prev_acc_lst = [0.] * 3
		prev_acc_test_lst = [0.] * 3

		for n in range(1, config.num_epochs + 1):

			if n > 1 and config.lr[n - 1] != config.lr[n - 2]:
				print("[LR decayed: {} => {}]".format(config.lr[n - 2], config.lr[n - 1]))
				for param_group in optimizer.param_groups:
					param_group["lr"] = config.lr[n - 1]

			net.train()

			ntr = 0.
			nva = 0.
			nte = 0.
			
			loss_tr = 0.
			loss_va = 0.
			loss_te = 0.

			corr_tr = [0., 0., 0.]
			corr_va = [0., 0., 0.]
			corr_te = [0., 0., 0.]

			loaders = [None] * 6

			max_len = len(trainset[0])

			for step in range(max_len):

				for i in range(config.num_subjects - 1):
					if step % len(trainset[i]) == 0:
						loaders[i] = iter(trainset[i])
				
				x_tr, y_tr, yid_tr = next(loaders[0])
				
				for i in range(1, config.num_subjects - 1):
					x_tmp, y_tmp, yid_tmp = next(loaders[i])

					x_tr = torch.cat((x_tr, x_tmp), dim=0)
					y_tr = torch.cat((y_tr, y_tmp), dim=0)
					yid_tr = torch.cat((yid_tr, yid_tmp), dim=0)

				x_tr, y_tr, yid_tr = x_tr.cuda(), y_tr.cuda(), yid_tr.cuda()
				optimizer.zero_grad()
				out, feat = net(x_tr)
				loss = criterion(out, y_tr)

				loss_total = loss
				
				loss_total.backward()
				optimizer.step()
				loss_tr += float(loss_total.item())
				
				correct_lst = utils.topk_correct(out, y_tr, topk=(1, 3, 5))

				for i in range(3):
					corr_tr[i] += correct_lst[i]

				ntr += float(y_tr.size(0))
				
			net.eval()
			
			with torch.no_grad():
				for i, data in enumerate(valset):
					x_va, y_va, yid_va = data[0].cuda(), data[1].cuda(), data[2].cuda()
					out, _ = net(x_va)
					loss = criterion(out, y_va)
					loss_total = loss
					loss_va += float(loss.item())
					
					correct_lst = utils.topk_correct(out, y_va, topk=(1, 3, 5))

					for i in range(3):
						corr_va[i] += correct_lst[i]
						
					nva += float(y_va.size(0))

				for i, data in enumerate(testset):
					x_te, y_te, yid_te= data[0].cuda(), data[1].cuda(), data[2].cuda()
					out, _ = net(x_te)
					loss = criterion(out, y_te)
					loss_total = loss
					loss_te += float(loss.item())
					
					correct_lst = utils.topk_correct(out, y_te, topk=(1, 3, 5))

					for i in range(3):
						corr_te[i] += correct_lst[i]
						
					nte += float(y_te.size(0))

				print('{:04d} epoch train loss: {:.4f} val loss: {:.4f}'.format(n, float(loss_tr/ntr), float(loss_va/nva)))
				print('\t[Class top-1] train acc:  {:.4f} val acc: {:.4f} test acc: {:.4f}' .format(corr_tr[0]/ntr, corr_va[0]/nva, corr_te[0]/nte))

				if prev_acc_lst[0] < corr_va[0] / nva:
					prev_acc_lst = [float(item) / nva for item in corr_va]
					prev_acc_test_lst = [float(item) / nva for item in corr_te]
					prev_epoch = n
					print('> Max validation accuracy: (top-1) {:.4f}, (top-3) {:.4f}, (top-5) {:.4f}'.format(prev_acc_lst[0], prev_acc_lst[1], prev_acc_lst[2]))

					torch.save(net.state_dict(), os.path.join(config.model_path, "best_model_split_{}.pkl".format(split_no)))
					utils.save_best_record(os.path.join(config.output_path, "best_score_split_{}.txt".format(split_no)), prev_epoch, prev_acc_lst, prev_acc_test_lst)

		end_time = time.time()
		elapsed = end_time - start_time

		print('Split no. {} finished'.format(split_no))
		
		print('[Best] Validation (top-1) {:.4f}, (top-3) {:.4f}, (top-5) {:.4f}'.format(prev_acc_lst[0], prev_acc_lst[1], prev_acc_lst[2]))
		print('[Best] Test (top-1) {:.4f}, (top-3) {:.4f}, (top-5) {:.4f}'.format(prev_acc_test_lst[0], prev_acc_test_lst[1], prev_acc_test_lst[2]))
		print('Total elapsed time: {} hours {} mins'.format(int(elapsed) // 3600, (int(elapsed) % 3600) // 60))
