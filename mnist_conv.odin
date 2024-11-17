package main

import "stml"
import "core:fmt"
import "core:time"
import "core:slice"
import alg "core:math/linalg"
import rl "vendor:raylib"

mnist_load :: proc($images_path, $labels_path: string, $count: int) -> (^stml.T(count, 28, 28), ^stml.M(count, 1)) {
	images := new(stml.T(count, 28, 28))
	labels := new(stml.M(count,      1))

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

// Load all model parametrs
c1w := stml.as( 4,1,5,5, #load("assets/conv_weights/conv1.weight.bin", []f32))
c2w := stml.as( 8,4,5,5, #load("assets/conv_weights/conv2.weight.bin", []f32))
c3w := stml.as(16,8,5,5, #load("assets/conv_weights/conv3.weight.bin", []f32))
fcw := stml.as(  10,144, #load("assets/conv_weights/fc.weight.bin", []f32))
c1b := stml.as(       4, #load("assets/conv_weights/conv1.bias.bin", []f32))
c2b := stml.as(       8, #load("assets/conv_weights/conv2.bias.bin", []f32))
c3b := stml.as(      16, #load("assets/conv_weights/conv3.bias.bin", []f32))
fcb := stml.as(      10, #load("assets/conv_weights/fc.bias.bin", []f32))

// Intermediate values
out1 := new(stml.T(4,24,24))
out2 := new(stml.T(8,10,10))
out3 := new(stml.T(16,3,3))
out  := new(stml.V(10))

// Textures for intermediate values
tex_out1: [ 4]rl.Texture
tex_out2: [ 8]rl.Texture
tex_out3: [16]rl.Texture

run :: proc(img: ^stml.T(1,28,28)) {
	using stml
	zero(out1); zero(out2); zero(out3); zero(out)
	conv2d(img, c1w, c1b, out1, stride=1)
	relu_(out1)
	conv2d(out1, c2w, c2b, out2, stride=2)
	relu_(out2)
	conv2d(out2, c3w, c3b, out3, stride=2)
	relu_(out3)
	mulT(as(1, 16*3*3, out3.data[:]), fcw, as(1, 10, out.data[:]))
	add(out, fcb, out)
	sigmoid_(out)
}

stml_to_rl :: proc(s: ^stml.M($x, $y), r: ^rl.Texture) {
	if r.id != 0 { rl.UnloadTexture(r^) }
	cp := make([]u8, x*y)
	defer delete(cp)

	for i in 0..<x*y { cp[i] = u8(clamp(s.data[i], 0, 1)*255) }
	r^ = rl.LoadTextureFromImage({
		data = raw_data(cp),
		width = i32(x),
		height = i32(y),
		mipmaps = 1,
		format = .UNCOMPRESSED_GRAYSCALE,
	})
}

draw_texture :: proc(r: rl.Texture, x, y: int, scale: f32) {
	src := rl.Rectangle{0,0,f32(r.width),f32(r.height)}
	dst := rl.Rectangle{f32(x), f32(y), f32(r.width)*scale, f32(r.height)*scale}
	rl.DrawTexturePro(r, src, dst, 0, 0, rl.WHITE)
}

main :: proc () {
	rl.SetTraceLogLevel(.WARNING)
	// rl.SetConfigFlags(rl.ConfigFlags{.VSYNC_HINT})
	rl.InitWindow(1000, 1000, "MNIST")
	rl.SetTargetFPS(rl.GetMonitorRefreshRate(rl.GetCurrentMonitor()))
	defer rl.CloseWindow()

	pixels: [28*28]f32
	img := rl.Image{&pixels, 28, 28, 1, .UNCOMPRESSED_GRAYSCALE}
	tex := rl.LoadTextureFromImage(img)

	images, labels := mnist_load(
		"./assets/MNIST/raw/t10k-images-idx3-ubyte",
		"./assets/MNIST/raw/t10k-labels-idx1-ubyte",
		count=10000
	)
	sel := 0

	for !rl.WindowShouldClose() {
		changed := false
		dst := rl.Rectangle{0, 20, 280, 280}
		if rl.IsMouseButtonDown(.LEFT) && rl.CheckCollisionPointRec(rl.GetMousePosition(), dst) {
			m := (rl.GetMousePosition() - [2]f32{dst.x, dst.y}) / [2]f32{dst.width, dst.height}
			for y in 0..<28 {
				for x in 0..<28 {
					c := [2]f32{f32(x)/27, f32(y)/27}
					dist: f32 = alg.distance(m, c)
					scaled := 1 - 1650 * alg.pow(dist, 2.5)
					scaled = clamp(scaled, 0, 1)
					pixels[y*28+x] = max(pixels[y*28+x], scaled)
					// u8pixels[y*28+x] = u8(pixels[y*28+x] * 255)
				}
			}
			changed = true
		}

		if rl.IsKeyPressed(.RIGHT) {
			sel = ((sel+1) + type_of(images).a) % type_of(images).a
			copy(pixels[:], stml.get(sel, images).data[:])
			changed = true
		}
		if rl.IsKeyPressed(.LEFT) {
			sel = ((sel-1) + type_of(images).a) % type_of(images).a
			copy(pixels[:], stml.get(sel, images).data[:])
			changed = true
		}

		if changed {
			run(stml.as(1,28,28, pixels[:]))
			stml_to_rl(stml.as(28,28, pixels[:]), &tex)
			for i in 0..< 4 { stml_to_rl(stml.get(i, out1), &tex_out1[i]) }
			for i in 0..< 8 { stml_to_rl(stml.get(i, out2), &tex_out2[i]) }
			for i in 0..<16 { stml_to_rl(stml.get(i, out3), &tex_out3[i]) }
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.Color{53,54,54,255})

		rl.DrawText("input", 0, 0, 20, rl.BLACK)
		rl.DrawTexturePro(tex, rl.Rectangle{0,0,28,28}, dst, 0, 0, rl.WHITE)

		rl.DrawText("layer 1", 0, 300, 20, rl.BLACK)
		for i in 0..< 4 { draw_texture(tex_out1[i], 60*i, 320, 2) }

		rl.DrawText("layer 2", 0, 400, 20, rl.BLACK)
		for i in 0..< 8 { draw_texture(tex_out2[i], 50*i, 420, 4) }

		rl.DrawText("layer 3", 0, 500, 20, rl.BLACK)
		for i in 0..<16 { draw_texture(tex_out3[i], 50*i, 520, 16) }

		for i in 0..<10 {
			rl.DrawText(fmt.ctprint(i), 0, i32(600+i*20), 20, rl.BLACK)
			rl.DrawRectangle(20, i32(600+i*20), i32(out.data[i]*100), 20, rl.BLACK)
		}

		rl.DrawFPS(rl.GetScreenWidth()-80,rl.GetScreenHeight()-20)
		rl.EndDrawing()
		free_all(context.temp_allocator)
	}

	// img0 := as(1, 28, 28, images.data[2*28*28:][:28*28])
	// imshow(as(28, 28, img0.data[:]))


	// fmt.println(out)
}
