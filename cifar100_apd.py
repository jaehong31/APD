import os
import tensorflow as tf
import numpy as np
import time
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
from ops import *
import pdb
from sklearn.metrics import roc_curve, auc
tdist = tf.contrib.distributions

class MODEL(object):
	def __init__(self, config, task_id, test=False):
		self.test=test
		self.tid = task_id
		self.c = config
		self.params = dict()
		self.czero = tf.constant(0, dtype=tf.float32)
		self.remask = False
		# NOTE After training, local_id += 1
		self.local_id = int(int((self.tid) / self.c.clustering_iter) * self.c.clustering_iter)

		for i in [0, 1]:
			#[5, 5, 3, 20], [5, 5, 20, 50]
			w = self.create_variable('global', 'shared_weights/layer%d'%i, [5, 5, self.c.dims[i], self.c.dims[i+1]])
			b = self.create_variable('task%d'%self.tid, 'biases/layer%d'%i, [self.c.dims[i+1]])
			print('%s, %s'%(w.shape, w.name))

			# NOTE Mask initialization
			m = self.create_variable('task%d'%self.tid, 'mask/layer%d'%i, self.c.dims[i+1])

		for i in [2, 3]:
			# NOTE transpose
			w = self.create_variable('global', 'shared_weights/layer%d'%i, [self.c.dims[i+1], self.c.dims[i+2]])
			b = self.create_variable('task%d'%self.tid, 'biases/layer%d'%i, [self.c.dims[i+2]])
			print('%s, %s'%(w.shape, w.name))

			# NOTE Mask initialization
			m = self.create_variable('task%d'%self.tid, 'mask/layer%d'%i, self.c.dims[i+2])


	def set_initial_states(self, decay_step):
		self.g_step = tf.Variable(0., trainable=False)
		self.lr = tf.train.exponential_decay(
					self.c.init_lr,           # Base learning rate.
					self.g_step*self.c.batch_size,  # Current index into the dataset.
					decay_step,          # Decay step.
					0.95,                # Decay rate.
					staircase=True)
		self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
		self.Y = tf.placeholder(tf.float32, [None, self.c.n_classes])

	def create_variable(self, scope, name, shape, trainable=True, add_param=True,
			initializer=tf.variance_scaling_initializer()):
		with tf.variable_scope(scope):
			if shape is not None:
				w = tf.get_variable(name, shape, trainable = trainable, initializer=initializer)
			else: # create using predifined parameter
				w = tf.get_variable(name, trainable = trainable, initializer=initializer)
			if add_param:
				self.params[w.name] = w.shape
		return w

	def get_variable(self, scope, name, trainable=True, reuse=True):
		with tf.variable_scope(scope, reuse=True):
			w = tf.get_variable(name, trainable=trainable)
			self.params[w.name] = w.shape
		return w

	def save(self, model_dir, option=None, remove_ploc=None):
		# NOTE remove prev_local_shared for efficiency
		if remove_ploc is not None:
			save_vars = [vv for vv in tf.trainable_variables() if not 'id%d_local'%remove_ploc in vv.name]
		else:
			save_vars = tf.trainable_variables()
		saver = tf.train.Saver(tf.trainable_variables())
		name = 'model_task%d_%s.ckpt'%(self.tid, option) if option else 'model_task%d.ckpt'%self.tid
		saver.save(self.sess, os.path.join(model_dir, name))


	def load_params(self, tid, model_dir, load_type='all', option=None):
		name = 'model_task%d_%s.ckpt'%(tid, option) if option else 'model_task%d.ckpt'%tid
		model_path = os.path.join(model_dir, name)
		print("[*] Restoring Learned model (task_id={}) ..., load_type is \"{}\", file name is {}".format(tid, load_type, name))

		def various_cond(var, cond):
			condA = False
			condB = False
			if 'prv_' in cond:
				condA = not 'task%d/'%self.tid in var.name
			if 'mask' in cond:
				condB = 'mask' in var.name
			elif 'aw' in cond:
				condB = 'aw_all' in var.name or 'aw_group' in var.name

			output = condB if not 'prv_' in cond else condA and condB
			return output

		if load_type=='all':
			orgs = tf.trainable_variables()
		elif load_type=='sw':
			orgs = [tv for tv in tf.trainable_variables() if 'shared_weights' in tv.name]
		elif load_type=='cores':
			orgs = [tv for tv in tf.trainable_variables() if 'biases' in tv.name or 'topmost' in tv.name]
		else:
			orgs = [tv for tv in tf.trainable_variables() if various_cond(tv, load_type)]

		saver = tf.train.Saver(orgs)
		saver.restore(self.sess, model_path)
		print("Load Done.")


	def destroy_graph(self):
		tf.reset_default_graph()


	def create_task_parameters(self, list):
		print('CREATE TASK PARAMETERS')
		for key in list.keys():
			_key = key.split(':')[0]
			key_split = _key.split('/', 1)
			try:
				_ = self.create_variable(key_split[0], key_split[1], None, True, add_param=False, initializer=tf.zeros(list[key]))
				print('variable[%s] is created.'%_key)
			except ValueError:
				pass
		print('DONE')


	def _conv(self, bottom, kernel, biases, idx):
		conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding = 'SAME')
		conv = tf.nn.relu(tf.nn.bias_add(conv, biases))
		norm = tf.nn.lrn(conv, 4, bias = 1.0, alpha = 0.001 / 9.0,
							beta = 0.75, name = 'norm%d'%idx)
		pool = tf.nn.max_pool(norm, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],
							padding = 'SAME', name = 'pool%d'%idx)
		return pool

	def gen_mask(self, mask):
		return tf.nn.sigmoid(mask)

	def l1_pruning(self, weights, hyp):
		hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
		return tf.multiply(weights, hard_threshold)

	def build_model_test(self, is_hierarchy=False, is_single_group=True):
		bottom, Y = self.X, self.Y
		n_total_elems, n_active_elems = 0, 0
		for i in [0, 1]:
			sw = self.get_variable('global', 'shared_weights/layer%d'%i, False, reuse=True)
			b = self.get_variable('task%d'%self.tid, 'biases/layer%d'%i, False, reuse=True)
			mask = self.get_variable('task%d'%self.tid, 'mask/layer%d'%i, False, reuse=True)
			g_mask = self.gen_mask(mask)

			# NOTE Tasks which are already accomplished k-means clustering
			if is_hierarchy and self.tid < self.local_id:
				group_info = self.assign_list[i][self.tid]

				# NOTE But if it is a single item in a group; there is no actual local-shared; pass this condition.
				is_single_group = np.sum(np.equal(self.assign_list[i], group_info)) == 1
				if is_single_group:
					pass
				else:
					local_shared = self.get_variable('id%d_local%d'%(self.local_id, group_info), 'aw_group/layer%d'%i, False, reuse=True)
					local_shared = self.l1_pruning(local_shared, self.c.l1_hyp)

					aw_mar = self.get_variable('task%d'%self.tid, 'aw_all/layer%d'%i, False, reuse=True)
					pruned_aw_mar = self.l1_pruning(aw_mar, self.c.l1_hyp)
					pruned_aw = local_shared + pruned_aw_mar
			else:
				pass
			if is_single_group:
				aw = self.get_variable('task%d'%self.tid, 'aw_all/layer%d'%i, False, reuse=True)
				pruned_aw = self.l1_pruning(aw, self.c.l1_hyp)
			else:
				pass

			pruned_m = self.l1_pruning(g_mask, self.c.mask_hyp)
			bottom = self._conv(bottom, sw * pruned_m + pruned_aw, b, i)
		flatten = bottom.get_shape().as_list()
		bottom = tf.reshape(bottom, [-1, np.prod(flatten[1:])])

		for i in [2, 3]:
			sw = self.get_variable('global', 'shared_weights/layer%d'%i, False, reuse=True)
			b = self.get_variable('task%d'%self.tid, 'biases/layer%d'%i, False, reuse=True)
			mask = self.get_variable('task%d'%self.tid, 'mask/layer%d'%i, False, reuse=True)
			g_mask = self.gen_mask(mask)

			if is_hierarchy and self.tid < self.local_id:
				group_info = self.assign_list[i][self.tid]
				is_single_group = np.sum(np.equal(self.assign_list[i], group_info)) == 1

				if is_single_group:
					pass
				else:
					local_shared = self.get_variable('id%d_local%d'%(self.local_id, group_info), 'aw_group/layer%d'%i, False, reuse=True)
					local_shared = self.l1_pruning(local_shared, self.c.l1_hyp)

					aw_mar = self.get_variable('task%d'%self.tid, 'aw_all/layer%d'%i, False, reuse=True)
					pruned_aw_mar = self.l1_pruning(aw_mar, self.c.l1_hyp)
					pruned_aw = local_shared + pruned_aw_mar
			else:
				pass

			if is_single_group:
				aw = self.get_variable('task%d'%self.tid, 'aw_all/layer%d'%i, False, reuse=True)
				pruned_aw = self.l1_pruning(aw, self.c.l1_hyp)
			else:
				pass

			pruned_m = self.l1_pruning(g_mask, self.c.mask_hyp)
			bottom = tf.nn.relu(tf.matmul(bottom, sw * pruned_m + pruned_aw) + b)

		prev_dim = bottom.get_shape().as_list()[1]
		topw = self.get_variable('task%d'%self.tid, 'topmost_weights', False, reuse=True)
		topb = self.get_variable('task%d'%self.tid, 'topmost_biases', False, reuse=True)

		y = tf.matmul(bottom, topw) + topb
		self.y_hat_test = tf.nn.softmax(y)


	def build_model(self, is_hierarchy=False, is_single_group=True):
		bottom, Y = self.X, self.Y
		weight_decay = 0
		sparseness, approx_loss = 0, 0
		for i in [0, 1]:
			sw = self.get_variable('global', 'shared_weights/layer%d'%i, True) # Trainable
			b = self.get_variable('task%d'%self.tid, 'biases/layer%d'%i, True)
			mask = self.get_variable('task%d'%self.tid, 'mask/layer%d'%i, True)
			g_mask = self.gen_mask(mask)

			# NOTE Initialize adaptive weights
			aw = self.create_variable('task%d'%self.tid, 'aw_all/layer%d'%i, None,
							True, initializer=sw)

			bottom = self._conv(bottom, sw * g_mask + aw, b, i)

			# NOTE Penalty Terms
			weight_decay += self.c.wd_rate * tf.nn.l2_loss(aw)
			weight_decay += self.c.wd_rate * tf.nn.l2_loss(mask)
			sparseness += self.c.l1_hyp * tf.reduce_sum(tf.abs(aw))
			sparseness += self.c.mask_hyp * tf.reduce_sum(tf.abs(mask))

			if self.tid == 0:
				weight_decay += self.c.wd_rate * tf.nn.l2_loss(sw)
			else:
				# NOTE Handle previous aw
				for j in range(self.tid):
					pmask = self.get_variable('task%d'%j, 'mask/layer%d'%i, self.remask, reuse=True)
					g_pmask = self.gen_mask(pmask)

					# NOTE Tasks which are already accomplished k-means clustering
					if is_hierarchy and j < self.local_id:
						group_info = self.assign_list[i][j]
						is_single_group = np.sum(np.equal(self.assign_list[i], group_info)) == 1

						# NOTE If it is a single item in a group; there is no actual local-shared; pass this condition.
						if not is_single_group:
							local_shared = self.get_variable('id%d_local%d'%(self.local_id, group_info), 'aw_group/layer%d'%i, False, reuse=True)
							paw = self.get_variable('task%d'%j, 'aw_all/layer%d'%i, True, reuse=True)
							full_paw = local_shared + paw

					# NOTE Not clustered yet
					elif j >= self.local_id:
						is_single_group = True
					else:
						pass
					if is_single_group:
						paw = self.get_variable('task%d'%j, 'aw_all/layer%d'%i, True, reuse=True)
						full_paw = paw

					theta_t = sw * g_pmask + full_paw
					a_l2 = tf.nn.l2_loss(theta_t - theta_t.eval())

					approx_loss += self.c.approx_hyp * a_l2
					sparseness += self.c.l1_hyp * tf.reduce_sum(tf.abs(paw))

					print(' [*] layer:%d, task:%d, is_hierarchy: %s, is_single_group: %s'%(i, j, is_hierarchy, is_single_group))


		flatten = bottom.get_shape().as_list()
		bottom = tf.reshape(bottom, [-1, np.prod(flatten[1:])])

		for i in [2, 3]:
			sw = self.get_variable('global', 'shared_weights/layer%d'%i, True)
			b = self.get_variable('task%d'%self.tid, 'biases/layer%d'%i, True)
			mask = self.get_variable('task%d'%self.tid, 'mask/layer%d'%i, True)
			g_mask = self.gen_mask(mask)

			# NOTE Initialize adaptive weights
			aw = self.create_variable('task%d'%self.tid, 'aw_all/layer%d'%i, None,
							True, initializer=sw)
			bottom = tf.nn.relu(tf.matmul(bottom, sw * g_mask + aw) + b)

			# NOTE Penalty Terms
			weight_decay += self.c.wd_rate * tf.nn.l2_loss(aw)
			weight_decay += self.c.wd_rate * tf.nn.l2_loss(mask)
			sparseness += self.c.l1_hyp * tf.reduce_sum(tf.abs(aw))
			sparseness += self.c.mask_hyp * tf.reduce_sum(tf.abs(mask))

			if self.tid == 0:
				weight_decay += self.c.wd_rate * tf.nn.l2_loss(sw)
			else:
				# NOTE Handle previous aw
				for j in range(self.tid):
					local_shared_list = []
					pmask = self.get_variable('task%d'%j, 'mask/layer%d'%i, self.remask, reuse=True)
					g_pmask = self.gen_mask(pmask)

					if is_hierarchy and j < self.local_id:
						group_info = self.assign_list[i][j]
						is_single_group = np.sum(np.equal(self.assign_list[i], group_info)) == 1

						if not is_single_group:
							local_shared = self.get_variable('id%d_local%d'%(self.local_id, group_info), 'aw_group/layer%d'%i, False, reuse=True)
							paw = self.get_variable('task%d'%j, 'aw_all/layer%d'%i, True, reuse=True)
							full_paw = local_shared + paw
							local_shared_list.append(local_shared)

					elif is_hierarchy and j >= self.local_id:
						is_single_group = True
					else:
						pass

					if is_single_group:
						paw = self.get_variable('task%d'%j, 'aw_all/layer%d'%i, True, reuse=True)
						full_paw = paw

					theta_t = sw * g_pmask + full_paw
					a_l2 = tf.nn.l2_loss(theta_t - theta_t.eval())

					approx_loss += self.c.approx_hyp * a_l2
					sparseness += self.c.l1_hyp * tf.reduce_sum(tf.abs(paw))
					print(' [*] layer:%d, task:%d, is_hierarchy: %s, is_single_group: %s'%(i, j, is_hierarchy, is_single_group))

		prev_dim = bottom.get_shape().as_list()[1]
		topw = self.create_variable('task%d'%self.tid, 'topmost_weights', [prev_dim, self.c.n_classes], True)
		topb = self.create_variable('task%d'%self.tid, 'topmost_biases', [self.c.n_classes], True)

		self.y = tf.matmul(bottom, topw) + topb
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, labels = Y))
		self.loss = self.cross_entropy + weight_decay
		self.loss += sparseness
		self.loss += approx_loss
		self.y_hat = tf.nn.softmax(self.y)


	def k_means_clustering(self, prv_cent = False, is_decomposed = False):
		assign_list = []
		get_cents = []
		_k = int(self.c.k_centroides * (self.tid+1) / self.c.clustering_iter)

		if hasattr(self, 'assign_list'):
			is_decomposed = True

		for i in range(4):
			full_aw = []
			only_aw = []
			for tid in range(self.tid+1):
				get_aw = self.get_variable('task%d'%tid, 'aw_all/layer%d'%i, False, reuse=True)
				only_aw.append(get_aw)

				if is_decomposed and len(self.assign_list[0]) > tid:
					local_id = self.assign_list[i][tid]
					# NOTE not single group
					if np.sum(local_id == np.array(self.assign_list[i])) > 1:
						local_shared = self.get_variable('id%d_local%d'%(len(self.assign_list[0]), local_id), 'aw_group/layer%d'%i, False, reuse=True)
						full_aw.append(get_aw + local_shared)
						print('layer%d,  local%d + task%d_aw'%(i, local_id, tid))
					else:
						full_aw.append(get_aw)
						print('layer%d,  single_group%d_task%d_aw'%(i, local_id, tid))
				else:
					full_aw.append(get_aw)
					print('layer%d,  single_task%d_aw'%(i, tid))

			rs_aws = tf.random_shuffle(full_aw)
			_slice = rs_aws.get_shape().as_list()
			_slice[0] = _k
			if prv_cent:
				#start from prv centroides
				slc = tf.slice(rs_aws, np.zeros_like(_slice), _slice)
				slc = self.sess.run(slc)
				n_prv_cents = prv_cent[-1][i].shape[0]
				slc[:n_prv_cents] = prv_cent[-1][i]
				centroides = tf.Variable(slc)
			else:
				centroides = tf.Variable(tf.slice(rs_aws, np.zeros_like(_slice), _slice))

			expanded_vectors = tf.expand_dims(full_aw, 0)
			expanded_centroides = tf.expand_dims(centroides, 1)

			dims = np.arange(len(full_aw[0].shape)) + 2
			assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), dims), 0)
			means = tf.concat([tf.reduce_mean(tf.gather(full_aw, tf.reshape(tf.where( tf.equal(assignments, _c)),[1,-1])), reduction_indices=[1]) for _c in range(_k)], 0)

			update_centroides = tf.assign(centroides, means)
			self.variable_initialization()
			for step in range(100):
			   _, centroid_values, assignment_values = self.sess.run([update_centroides, centroides, assignments])

			print(' [*] k-means clustering of layer %d : %s'%(i, assignment_values.tolist()))
			self.aw_consolidation(_k, i, np.array(full_aw), np.array(only_aw), assignment_values)
			assign_list.append(assignment_values.tolist())
			get_cents.append(centroid_values)
		self.assign_list = assign_list
		return get_cents


	def aw_consolidation(self, _k, layer_id, full_aws, only_aws, cluster_info):
		op_list = []
		for _c in range(_k):
			is_group = cluster_info == _c
			group_aws = full_aws[is_group]

			if np.sum(is_group) == 1:
				print(' [*] #%d single group'%_c)
				full_single_aw = group_aws[0]
				only_single_aw = only_aws[is_group][0]

				if full_single_aw.name != only_single_aw.name:
					print(' [*] Single but different! %s <=> %s'%(full_single_aw.name, only_single_aw.name))
					op_list.append(only_single_aw.assign(full_single_aw))

			else:
				e_max = tf.reduce_max(group_aws.tolist(), 0)
				e_min = tf.reduce_min(group_aws.tolist(), 0)

				e_gap = e_max-e_min
				e_nind = tf.cast(tf.greater(e_gap, self.c.e_gap_hyp), tf.float32)
				e_ind = tf.abs(1-e_nind)

				new_aw = self.create_variable('id%d_local%d'%(self.tid+1, _c), 'aw_group/layer%d'%layer_id,
												None, True, initializer =  tf.reduce_mean([e_max, e_min], 0) * e_ind)

				local_capacity = tf.reduce_sum(tf.cast(tf.not_equal(e_ind, tf.zeros_like(e_ind)), tf.int32))
				print(' [*] #%d local_shared_elements: %d/%d'%(_c, self.sess.run(local_capacity), np.prod(new_aw.get_shape().as_list())))

				for _f_aws, _o_aws in zip(group_aws.tolist(), only_aws[is_group].tolist()):
					op_list.append(_o_aws.assign(_f_aws * e_nind))

		self.variable_initialization()
		self.sess.run(op_list)
		print(' [*] consolidation end')

	def get_performance(self, p, y, measure='auc'):
		perf_list = []
		if measure == 'auc':
			for _i in range(self.c.n_classes):
				roc, perf = ROC_AUC(p[:,_i], y[:,_i])
				perf_list.append(perf)
		elif measure == 'acc':
			perf_list.append(accuracy(p, y))
		else:
			print('ERROR, get_performance measure is wrong. -> %s'%measure)
			sys.exit()
		return np.mean(perf_list)



	def get_next_batch(self, X, Y, is_train=True):
		if is_train:
			num_train = len(Y)
			if not hasattr(self, 'data_range'):
				self.epoch_idx = 0
				self.data_range = np.array(range(num_train))
				random.shuffle(self.data_range)
				self.trainX_shuffle = X[self.data_range]
				self.trainY_shuffle = Y[self.data_range]

			images_batch = np.zeros([0, 32, 32, 3], dtype=np.float32)
			labels_batch = np.zeros([0, self.c.n_classes], dtype=np.int32)
			batch_cnt = 0
			while True:
				if self.epoch_idx + (self.c.batch_size - batch_cnt) > num_train:
					temp_cnt = num_train - self.epoch_idx
				else:
					temp_cnt = self.c.batch_size - batch_cnt
				images_batch = np.concatenate([images_batch, self.trainX_shuffle[self.epoch_idx:self.epoch_idx+temp_cnt]], axis=0)
				labels_batch = np.concatenate([labels_batch, self.trainY_shuffle[self.epoch_idx:self.epoch_idx+temp_cnt]], axis=0)
				self.epoch_idx += temp_cnt
				if self.epoch_idx == num_train:
					self.epoch_idx = 0
					random.shuffle(self.data_range)
					self.trainX_shuffle = X[self.data_range]
					self.trainY_shuffle = Y[self.data_range]

				batch_cnt += temp_cnt
				if batch_cnt == self.c.batch_size:
					break
			return images_batch, labels_batch
		else:
			return X, Y


	def variable_initialization(self):
		'''Initialize only uninitialized (= newly added) variables.'''
		uninitialized_vars = []
		for var in tf.global_variables():
			try:
				self.sess.run(var)
			except tf.errors.FailedPreconditionError:
				uninitialized_vars.append(var)

		init_new_vars_op = tf.initialize_variables(uninitialized_vars)
		self.sess.run(init_new_vars_op)

	def optimization(self):
		print("[*] Starting optimization...")
		# NOTE TRAINING ONLY
		if not self.test:
			opt = tf.train.AdamOptimizer(self.lr, name='task%d/adam'%self.tid)
			asd = [vv for vv in tf.trainable_variables() if not 'local' in vv.name]
			grads_and_vars = opt.compute_gradients(self.loss, [vv for vv in tf.trainable_variables() if not 'local' in vv.name])
			for ind, (grad, var) in enumerate(grads_and_vars):
				try:
					grad = grad * 10.
					grads_and_vars[ind] = (grad, var)
				except TypeError:
					pass
			self.apply_grads = opt.apply_gradients(grads_and_vars, global_step=self.g_step)
		self.variable_initialization()

	def run(self, data):
		train, val, test = data
		train_data, train_labels = zip(*train)
		val_data, val_labels = zip(*val)
		test_data, test_labels = zip(*test)
		n_train_steps = int(len(train_data) / self.c.batch_size)
		n_val_steps = int(len(val_data) / self.c.batch_size)
		n_test_steps = int(len(test_data) / self.c.batch_size)

		train_data = np.array(train_data)
		train_labels = np.array(train_labels)
		val_data = np.array(val_data)
		val_labels = np.array(val_labels)
		test_data = np.array(test_data)
		test_labels = np.array(test_labels)

		val_list = []
		# NOTE TRAINING ONLY
		if not self.test:
			for epoch in range(self.c.n_epochs):
				start_time = time.time()
				train_costs, train_accs = [], []
				for _ in range(0, n_train_steps):
					train_X, train_Y = self.get_next_batch(train_data, train_labels)
					cost_, y_hat, lrn_rate_, _ = self.sess.run([self.loss, self.y_hat, self.lr, self.apply_grads],
						feed_dict={self.X: train_X, self.Y: train_Y})
					auc_ = self.get_performance(y_hat, train_Y, measure='acc')
					train_costs.append(cost_)
					train_accs.append(auc_)

				# TODO ITERATION IS REQUIRED?
				vcost_, vyhat_ = self.sess.run([self.loss, self.y_hat], feed_dict={self.X: val_data, self.Y: val_labels})
				val_perf = self.get_performance(vyhat_, np.array(val_labels), measure='acc')
				val_list.append(val_perf)


				if (epoch+1) + 5 >= self.c.n_epochs:
					self.save(self.c.results_path, option='epoch%d'%epoch)

				# Test
				test_accs = []
				y_hat = self.sess.run(self.y_hat_test, feed_dict={self.X: test_data})
				auc_ = self.get_performance(y_hat, np.array(test_labels), measure='acc')
				test_accs.append(auc_)

				print("{}th epoch, train_costs: {:.4f}, train_acc: {:.4f}, val_acc:{:.4f}, test_acc:{:.4f}, {:.3f} seconds, lrn_rate:{:.5f}". \
						format(epoch+1, np.mean(train_costs), np.mean(train_accs), val_list[-1], np.mean(test_accs), time.time()-start_time, lrn_rate_))

			return val_list
		else:
			test_accs = []
			y_hat = self.sess.run(self.y_hat_test, feed_dict={self.X: test_data})
			auc_ = self.get_performance(y_hat, test_labels, measure='acc')
			test_accs.append(auc_)

			print("task_id {:d}: final_test_acc:{:.4f}". format(self.tid, np.mean(test_accs)))
			return np.mean(test_accs)
