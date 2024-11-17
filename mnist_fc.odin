package main

import "stml"
import "core:fmt"
import "core:time"

load :: proc($r, $c: int, $path: string) -> ^stml.M(r,c) {
	bytes := #load(path)
	return transmute(^stml.M(r, c))raw_data(bytes)
}

mnist_load :: proc($images_path, $labels_path: string, $count: int) -> (^stml.M(count, 28*28), ^stml.M(count, 1)) {
	images := new(stml.M(count, 28*28))
	labels := new(stml.M(count,     1))

	_img_bytes := #load(images_path)
	img_bytes := _img_bytes[16:]
	for i in 0..<len(img_bytes) {
		images.data[i] = f32(img_bytes[i]) / 255.0
	}

	_lbl_bytes := #load(labels_path)
	lbl_bytes := _lbl_bytes[8:]
	for i in 0..<len(lbl_bytes) {
		labels.data[i] = f32(lbl_bytes[i])
	}

	return images, labels
}

main :: proc () {
	// q := new(stml.Q(1,2,3,4))
	// stml.test(q)
	// b := stml._test(1, q)
	// fmt.println(b)

	img := new(stml.T(1, 28, 28))
	w := new(stml.Q(4, 1, 5, 5))
	b := new(stml.V(4))
	out := new(stml.T(4, 24, 24))
	stml.conv2d(img, w, b, out)


	/*
	fc1_weight := load(20, 784, "./assets/fc_weights/fc1.weight.bin")
	fc1_bias   := load( 1,  20, "./assets/fc_weights/fc1.bias.bin")
	fc2_weight := load(10,  20, "./assets/fc_weights/fc2.weight.bin")
	fc2_bias   := load( 1,  10, "./assets/fc_weights/fc2.bias.bin")

	images, labels := mnist_load(
		"./assets/MNIST/raw/t10k-images-idx3-ubyte",
		"./assets/MNIST/raw/t10k-labels-idx1-ubyte",
		count=60000
	)

	img := stml.reshape(images.data[:28*28], 1, 28*28)
	stml.imshow(stml.reshape(img, 28, 28))

	hid1 := new(stml.M(1, 20))
	hid2 := new(stml.M(1, 10))

	stml.mulT(img,  fc1_weight, hid1)
	stml.add(hid1,  fc1_bias,   hid1)
	stml.relu_(hid1)
	stml.mulT(hid1, fc2_weight, hid2)
	stml.add(hid2,  fc2_bias,   hid2)
	stml.sigmoid_(hid2)

	stml.print(hid2)
	*/
}
