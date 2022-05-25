# Data Loader
import numpy as np
import scipy.io as sio
import pdb
import pickle
import random
import tensorflow as tf

def cifar100_python(data_path, group=1, validation=False, val_ratio=0.2, flat=False, one_hot=True):
	n_classes = 100

	files = open(data_path, 'rb')
	dict = pickle.load(files, encoding='bytes')

	# NOTE Image Standardization
	images = (dict[b'data'])
	images = np.float32(images)/255
	labels = dict[b'fine_labels']

	sort_l = np.sort(labels)
	argsort_l = np.argsort(labels)

	train_split = []
	val_split = []
	prv_position = 0
	# If group=10, [10, 20, 30, ..., 100]
	for idx in range(group, n_classes+1, int(n_classes/group)):
		position = sort_l.tolist().index(idx) if idx < n_classes else len(sort_l)
		print('range : [%d,%d]'%(prv_position, position))
		gimages = np.take(images,argsort_l[prv_position:position], axis=0)
		if not flat:
			gimages = gimages.reshape([gimages.shape[0], 32, 32, 3])
			#gimages = tf.image.per_image_standardization(gimages)


		glabels = np.take(labels,argsort_l[prv_position:position])
		if one_hot:
			# NOTE Edit label id to be in [0,9]. Each of task is a 10 classes problem.
			glabels = np.eye(int(n_classes/group))[glabels-(idx-10)]

		pairs = list(zip(gimages, glabels))
		random.shuffle(pairs)
		#gimages, glabels = zip(*pairs)
		if validation:
			spl = int(len(pairs)*(1-val_ratio))
			train_split.append(pairs[:spl])
			val_split.append(pairs[spl:])

		else:
			train_split.append(pairs)
		prv_position = position

	output = (train_split, val_split) if validation else train_split
	return output

def cifar100_superclass_python(data_path, group=5, validation=False, val_ratio=0.2, flat=False, one_hot=True):
	CIFAR100_LABELS_LIST = [
		'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
		'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
		'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
		'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
		'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
		'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
		'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
		'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
		'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
		'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
		'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
		'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
		'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
		'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
		'worm'
	]
	sclass = []
	sclass.append(' beaver, dolphin, otter, seal, whale,') 						#aquatic mammals
	sclass.append(' aquarium_fish, flatfish, ray, shark, trout,')				#fish
	sclass.append(' orchid, poppy, rose, sunflower, tulip,')					#flowers
	sclass.append(' bottle, bowl, can, cup, plate,')							#food
	sclass.append(' apple, mushroom, orange, pear, sweet_pepper,')				#fruit and vegetables
	sclass.append(' clock, computer keyboard, lamp, telephone, television,')	#household electrical devices
	sclass.append(' bed, chair, couch, table, wardrobe,')						#household furniture
	sclass.append(' bee, beetle, butterfly, caterpillar, cockroach,')			#insects
	sclass.append(' bear, leopard, lion, tiger, wolf,')							#large carnivores
	sclass.append(' bridge, castle, house, road, skyscraper,')					#large man-made outdoor things
	sclass.append(' cloud, forest, mountain, plain, sea,')						#large natural outdoor scenes
	sclass.append(' camel, cattle, chimpanzee, elephant, kangaroo,')			#large omnivores and herbivores
	sclass.append(' fox, porcupine, possum, raccoon, skunk,')					#medium-sized mammals
	sclass.append(' crab, lobster, snail, spider, worm,')						#non-insect invertebrates
	sclass.append(' baby, boy, girl, man, woman,')								#people
	sclass.append(' crocodile, dinosaur, lizard, snake, turtle,')				#reptiles
	sclass.append(' hamster, mouse, rabbit, shrew, squirrel,')					#small mammals
	sclass.append(' maple_tree, oak_tree, palm_tree, pine_tree, willow_tree,')	#trees
	sclass.append(' bicycle, bus, motorcycle, pickup_truck, train,')			#vehicles 1
	sclass.append(' lawn_mower, rocket, streetcar, tank, tractor,')				#vehicles 2

	n_classes = 100

	files = open(data_path, 'rb')
	dict = pickle.load(files, encoding='bytes')

	# NOTE Image Standardization
	images = (dict[b'data'])
	images = np.float32(images)/255
	labels = dict[b'fine_labels']
	labels_pair = [[jj for jj in range(100) if ' %s,'%CIFAR100_LABELS_LIST[jj] in sclass[kk]] for kk in range(20)]

	#flat_pair = np.concatenate(labels_pair)

	argsort_sup = [[] for _ in range(20)]
	for _i in range(len(images)):
		for _j in range(20):
			if labels[_i] in labels_pair[_j]:
				argsort_sup[_j].append(_i)

	argsort_sup_c = np.concatenate(argsort_sup)


	train_split = []
	val_split = []
	position = [_k for _k in range(0,len(images)+1,int(len(images)/20))]
	for idx in range(20):
		print('range : [%d,%d]'%(position[idx], position[idx+1]))
		gimages = np.take(images,argsort_sup_c[position[idx]:position[idx+1]], axis=0)

		if not flat:
			gimages = gimages.reshape([gimages.shape[0], 32, 32, 3])
			#gimages = tf.image.per_image_standardization(gimages)

		glabels = np.take(labels,argsort_sup_c[position[idx]:position[idx+1]])
		for _si, swap in enumerate(labels_pair[idx]):
			glabels = ['%d'%_si if x==swap else x for x in glabels]

		oh_labels = np.eye(group)[np.int32(glabels)]

		pairs = list(zip(gimages, oh_labels))
		random.shuffle(pairs)
		#gimages, glabels = zip(*pairs)
		if validation:
			spl = int(len(pairs)*(1-val_ratio))
			train_split.append(pairs[:spl])
			val_split.append(pairs[spl:])

		else:
			train_split.append(pairs)

	output = (train_split, val_split) if validation else train_split
	return output



def omniglot(path, n_tasks, n_classes, is_rotation=True, train=15):
	"""
	OUTPUT:	
		is_rotation==False
			shape of data : 1200 * 20 * Image_dim (n_classes * n_instances * image)
		is_rotation==True
			shape of data : 4800 * 20 * Image_dim ((n_classes * 4) * n_instances * image)
			=> Image1(0'), Image1(90'), Image1(180'), Image1(270'), Image2(0'), ...
	
 	USAGE EXAMPLE: 	
		all_data = data_loader.omniglot('/st1/jaehong/datasets/omniglot_anyshot/omni_train_rot.npy', n_tasks=FLAGS.n_tasks, n_classes=FLAGS.n_classes, train=15)

		for idx, t in enumerate(FLAGS.task_order[FLAGS.order_type][:FLAGS.n_tasks]):
			argdict = vars(FLAGS)
			print(argdict)
			data = (all_data[0][t], all_data[1][t], all_data[2][t], all_data[3][t])
			model = omniglot.MODEL(FLAGS, task_id=t)
		...
 
	"""
	data = np.load(path)

	n_angles = 4 if is_rotation else 1
	n_unique_images_per_task = n_classes * n_angles
	tr_n_eq_classes = train * n_angles
	te_n_eq_classes = (20-train) * n_angles

	_count = 0
	tr_images_list = []
	tr_labels_list = []
	te_images_list = []
	te_labels_list = []

	tr_labels = [(np.arange(n_classes) == float(int(j/tr_n_eq_classes))).astype(np.float32) for j in range(n_classes * tr_n_eq_classes)]
	te_labels = [(np.arange(n_classes) == float(int(j/te_n_eq_classes))).astype(np.float32) for j in range(n_classes * te_n_eq_classes)]
	
	for i in range(n_tasks):
		images = data[n_unique_images_per_task*i:n_unique_images_per_task*(i+1),:]

		tr_collect_im = images[:,:train].reshape([n_unique_images_per_task*train, 28, 28, 1])
		tr_images_list.append(tr_collect_im)
		tr_labels_list.append(tr_labels)
		te_collect_im = images[:,train:].reshape([n_unique_images_per_task*(20-train), 28, 28, 1])
		te_images_list.append(te_collect_im)
		te_labels_list.append(te_labels)
		_count += 1

	return tr_images_list, tr_labels_list, te_images_list, te_labels_list