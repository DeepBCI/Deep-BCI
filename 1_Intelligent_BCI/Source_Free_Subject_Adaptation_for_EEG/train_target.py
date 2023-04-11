
import numpy as np
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import *
from model import *
from model_generator import Generator
from options import *
from config import *
from contrastive import inter_subject_contrastive_loss
from mmd import MaximumMeanDiscrepancy
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

		trainset_data = eegloader(os.path.join(config.data_path, config.data_file), os.path.join(config.data_path, config.split_file), dtype='train', split_no=split_no, seed=config.seed, subject_id=config.target_subject, target_subject=config.target_subject, k_shot=config.k)

		batch_size = config.k * config.num_classes
		trainset = DataLoader(trainset_data, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)

		valset_data = eegloader(os.path.join(config.data_path, config.data_file), os.path.join(config.data_path, config.split_file), dtype='val', split_no=split_no, seed=config.seed, subject_id=config.target_subject, target_subject=config.target_subject, k_shot=config.k)
		testset_data = eegloader(os.path.join(config.data_path, config.data_file), os.path.join(config.data_path, config.split_file), dtype='test', split_no=split_no, seed=config.seed, subject_id=config.target_subject, target_subject=config.target_subject, k_shot=config.k)

		print(len(valset_data))
		print(len(testset_data))

		valset = DataLoader(valset_data, batch_size=config.batch_size_each, worker_init_fn=worker_init_fn, num_workers=4)
		testset = DataLoader(testset_data, batch_size=config.batch_size_each, worker_init_fn=worker_init_fn, num_workers=4)

		net = Model(config.num_classes, hidden_dim=128).cuda()

		criterion = nn.CrossEntropyLoss().cuda()
		optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0])
		print(net)

		g = Generator(input_dim=100, output_dim=128, hidden_dim=128, num_classes=40).cuda()

		ckpt_path_g = os.path.join(config.ckpt_path_g, 'best_model_split_{}.pkl'.format(split_no))

		g.load_state_dict(torch.load(ckpt_path_g))
		g.eval()
		g.requires_grad_(False)

		if args.mmd:
			mmd = MaximumMeanDiscrepancy(kernel_type='poly', normalize=True)

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

			loss_tr_contrast = 0.
			loss_tr_mmd = 0.

			corr_tr = [0., 0., 0.]
			corr_va = [0., 0., 0.]
			corr_te = [0., 0., 0.]

			loaders = None

			max_len = len(trainset)
			num_samples = config.batch_size_each * 5

			for step in range(max_len):

				if step % len(trainset) == 0:
					loaders = iter(trainset)
				
				x_tr, y_tr, yid_tr = next(loaders)

				num_samples_target = x_tr.size(0)
				
				# oversampling for the target subject
				over_multiplier = int(np.ceil(config.batch_size_each / x_tr.size(0)))

				x_tr = x_tr.repeat((over_multiplier, 1, 1))[:config.batch_size_each].cuda()
				y_tr = y_tr.repeat((over_multiplier,))[:config.batch_size_each].cuda()
				yid_tr = yid_tr.repeat((over_multiplier,))[:config.batch_size_each].cuda()

				out, feat = net(x_tr)

				z = torch.randn((num_samples_target * 5, 100)).cuda()
				y_tmp_g = torch.linspace(0, config.num_classes-1, steps=config.num_classes).long()
				y_tmp_g = torch.cat([y_tmp_g] * (num_samples_target * 5 // config.num_classes), dim=0).cuda()
				yid_tmp_g = torch.ones((num_samples_target * 5)).cuda() * -1

				x_tmp_g = g(z, y_tmp_g)

				optimizer.zero_grad()

				out_tmp_g = net.classifier(x_tmp_g)

				out_full = torch.cat((out, out_tmp_g), dim=0)
				y_tr_full = torch.cat((y_tr, y_tmp_g), dim=0)

				loss = criterion(out_full, y_tr_full)

				loss_total = loss

				if args.mmd:
					dist = mmd(feat[:num_samples_target], x_tmp_g)

					loss_mmd = dist
					loss_total += loss_mmd
				
				if args.contrastive:
					feat_full = torch.cat((feat, x_tmp_g), dim=0)
					y_tr_full = torch.cat((y_tr, y_tmp_g), dim=0)
					yid_tr_full = torch.cat((yid_tr, yid_tmp_g), dim=0)

					loss_contrastive = inter_subject_contrastive_loss(feat_full, y_tr_full, yid_tr_full, tau=config.tau)
					loss_total += loss_contrastive

				loss_total.backward()
				optimizer.step()
				loss_tr += float(loss_total.item())
				
				if args.mmd:
					loss_tr_mmd += float(loss_mmd.item())
				if args.contrastive:
					loss_tr_contrast += float(loss_contrastive.item())
				
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

				if args.contrastive:
					print('{:04d} epoch train loss: {:.4f} train loss (contrast): {:.4f} val loss: {:.4f}'.format(n, float(loss_tr/ntr), float(loss_tr_contrast/ntr), float(loss_va/nva)))
				elif args.mmd:
					print('{:04d} epoch train loss: {:.4f} train loss (mmd): {:.4f} val loss: {:.4f}'.format(n, float(loss_tr/ntr), float(loss_tr_mmd/ntr), float(loss_va/nva)))
				else:
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
