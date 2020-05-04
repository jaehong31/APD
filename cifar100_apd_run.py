"""
oracle with sw mask
"""

import os
import tensorflow as tf
import numpy as np
import argparse
import copy
import data_loader
import pdb

np.random.seed(1004)
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--n_epochs', type=int, default=20, help='the maximum number of epochs')
parser.add_argument('--decay_rate', type=int, default=1)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_tasks', type=int, default=10)
parser.add_argument('--init_lr', type=float, default=1e-3)
parser.add_argument('--wd_rate', type=float, default=1e-4)
parser.add_argument('--gpu_num', type=str, default="0")
parser.add_argument('--results_path', type=str, default='./results/apd/')
parser.add_argument('--memory_usage', type=float, default=0.95)

parser.add_argument('--mask_hyp', type=float, default=0)
parser.add_argument('--approx_hyp', type=float, default=0)
parser.add_argument('--l1_hyp', type=float, default=0)

parser.add_argument('--order_type', type=str, default='orderA', choices=['orderA', 'orderB', 'orderC', 'orderD', 'orderE'])
parser.add_argument('--data_type', type=str, default='default', choices=['defailt', 'superclass'])

parser.add_argument('--e_gap_hyp', type=float, default=1e-4)
parser.add_argument('--clustering_iter', type=int, default=5)
parser.add_argument('--k_centroides', type=int, default=2)

FLAGS = parser.parse_args()
import cifar100_apd as cifar100

FLAGS.dims = [3, 20, 50, 3200, 800, 500, 10]
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_num

is_hier = False
avg_perf = []
min_vc_epochs = []
params = dict()
assign_list = [ [], [], [], [] ]
cent_list = []
if FLAGS.data_type == 'default':
	train_data, val_data = data_loader.cifar100_python('./cifar-100-python/train', group=FLAGS.n_classes, validation=True)
	test_data = data_loader.cifar100_python('./cifar-100-python/test', group=FLAGS.n_classes)

	FLAGS.task_order= {
			'orderA': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
			'orderB': [1, 7, 4, 5, 2, 0, 8, 6, 9, 3],
			'orderC': [7, 0, 5, 1, 8, 4, 3, 6, 2, 9],
			'orderD': [5, 8, 2, 9, 0, 4, 3, 7, 6, 1],
			'orderE': [2, 9, 5, 4, 8, 0, 6, 1, 3, 7]}


elif FLAGS.data_type == 'superclass':
	train_data, val_data = data_loader.cifar100_superclass_python('./cifar-100-python/train', group=FLAGS.n_classes, validation=True)
	test_data = data_loader.cifar100_superclass_python('./cifar-100-python/test', group=FLAGS.n_classes)

	FLAGS.task_order= {
			'orderA': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
			'orderB': [15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19],
			'orderC': [17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8],
			'orderD': [11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17],
			'orderE': [6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16]}


for idx, _t in enumerate(FLAGS.task_order[FLAGS.order_type][:FLAGS.n_tasks]):
	full_base = (5*5*3*20 + 5*5*20*50 + 3200*800 + 800*500 + 500*FLAGS.n_classes * idx)
	argdict = vars(FLAGS)
	print(argdict)
	active_m_sum, total_m_sum = 0, 0

	data = (train_data[_t], val_data[_t], test_data[_t])
	model = cifar100.MODEL(FLAGS, task_id=idx)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage)
	model.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

	model.set_initial_states(len(train_data[_t])*FLAGS.decay_rate)
	print("\n\n\tTASK %d TRAINING\n"%(idx))

	# NOTE If (idx+1) > clustering_iter; then there are local_shared parameters.
	if (idx+1 > FLAGS.clustering_iter):
		is_hier=True

	# NOTE If hierarchy; load model.assign_list
	if is_hier:
		model.assign_list = assign_list

	model_dir = FLAGS.results_path
	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)

	if idx != 0:
		prvs = {pv:params[pv] for pv in params.keys() if 'aw_' in pv or 'mask' in pv}
		model.create_task_parameters(prvs)

		model.load_params(tid=idx-1, model_dir=FLAGS.results_path, load_type='prv_mask', option='min_vcost')
		model.load_params(tid=idx-1, model_dir=FLAGS.results_path, load_type='prv_aw', option='min_vcost')
		model.load_params(tid=idx-1, model_dir=FLAGS.results_path, load_type='sw', option='min_vcost')
		model.variable_initialization()

	model.build_model(is_hier)
	model.build_model_test(is_hier)
	model.optimization()
	vcost_list = model.run(data)

	max_vperf = np.max(vcost_list[-5:])
	min_vc_epoch = vcost_list[-5:].index(max_vperf) + FLAGS.n_epochs - 5
	min_vc_epochs.append(min_vc_epoch)
	print('MIN VALIDATION COSTS(EPOCHS) at EACH TASKS: %s'%min_vc_epochs)

	model.load_params(tid=idx, model_dir=FLAGS.results_path, load_type='all', option='epoch%d'%min_vc_epoch) # restore all params

	# NOTE At each clustering_iter, do k_means_clustering and copy assignment information.
	if (idx+1) % FLAGS.clustering_iter == 0:
		get_centroids = model.k_means_clustering(cent_list)
		cent_list.append(get_centroids)
		assign_list = copy.copy(model.assign_list)


	print(' [*] After training, show parameter info.')
	# NOTE Remove temporal parameters which are from k_means_clustering.
	for vv in [_v for _v in tf.trainable_variables() if not 'Variable' in _v.name]:
		# NOTE Only aws are sparse.
		if 'aw' in vv.name:
			vvv = model.l1_pruning(vv, FLAGS.l1_hyp)
		else:
			vvv=vv

		# NOTE Compute active parameters (capacity)
		actives = tf.not_equal(vvv, tf.zeros_like(vvv))
		active_m = model.sess.run(tf.reduce_sum(tf.cast(actives, tf.float32)))
		total_m = np.prod(vv.get_shape().as_list())
		active_m_sum += active_m
		total_m_sum += total_m

	# NOTE Store parameter.name to load them after graph destory
	keys = model.params.keys()
	for key in keys:
		params[key] = copy.copy(model.params[key])

	# NOTE local_id: recently generated local_shared parameter id
	local_id = int(int((idx+1) / FLAGS.clustering_iter) * FLAGS.clustering_iter)
	# NOTE Remove old local-shared from checkpoints and params if new local-shared are generated
	if (idx+1) % FLAGS.clustering_iter == 0 and (idx+1) > FLAGS.clustering_iter:
		model.save(FLAGS.results_path, option='min_vcost', remove_ploc=int(idx+1-FLAGS.clustering_iter))

		remove_ploc_name = [pv for pv in params.keys() if 'id%d_local'%(local_id-FLAGS.clustering_iter) in pv]
		for _name in remove_ploc_name:
			print(' [*] old local-shared is deleted from params. %s'%_name)
			del params[_name]
	else:
		model.save(FLAGS.results_path, option='min_vcost')

	model.destroy_graph()
	model.sess.close()

	# NOTE TEST
	avg_aucs = []
	aucs = []
	updated_memory = 0.

	for iidx, _i in enumerate(FLAGS.task_order[FLAGS.order_type][:(idx+1)]):
		# NOTE potential to be hierarchical
		data = (train_data[_i], val_data[_i], test_data[_i])
		model = cifar100.MODEL(FLAGS, task_id=iidx, test=True)
		model.local_id = local_id
		model.sess = tf.Session()
		model.set_initial_states(len(train_data[_i]))
		print("\n\n\tTASK %d TEST\n"%(iidx))

		# NOTE when after first consolidation, identify newly introduced adaptive weights (new tasks) which is not consolidated yet.
		if ((idx+1) >= FLAGS.clustering_iter) and (local_id >= (iidx+1)):
			is_hier = True
			model.assign_list = assign_list
		else:
			is_hier = False

		i_adaptive = {ia:params[ia] for ia in params.keys() if 'task%d/'%iidx in ia or 'id%d_local'%local_id in ia}
		model.create_task_parameters(i_adaptive) # create required params to restore
		model.load_params(tid=idx, model_dir=FLAGS.results_path, load_type='aw', option='min_vcost')
		model.load_params(tid=idx, model_dir=FLAGS.results_path, load_type='mask', option='min_vcost')
		model.load_params(tid=iidx, model_dir=FLAGS.results_path, load_type='cores', option='min_vcost')#'epoch%d'%min_vc_epoch) # restore task-adaptive biases, topmost weights
		model.load_params(tid=idx, model_dir=FLAGS.results_path, load_type='sw', option='min_vcost') # restore sw

		model.build_model_test(is_hier)
		model.variable_initialization()
		auc = model.run(data)

		aucs.append('%.4f'%auc)
		avg_aucs.append('%.4f'%(np.mean(np.float32(aucs))))
		model.destroy_graph()
		model.sess.close()

	print('ACCS: %s'%aucs)
	print('AVGS: %s'%avg_aucs)
	print('Updated Memory Requirements: %.3f' %(active_m_sum/full_base))
	print('Memory Info: (%d/%d)' %(active_m_sum, full_base))
