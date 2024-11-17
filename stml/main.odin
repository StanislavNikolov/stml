package stml

import "core:fmt"
import "core:math/linalg"


/**************
* Basic types *
***************/

V :: struct($a: int)             { data: [a      ]f32 } // 1D - Vector
M :: struct($a, $b: int)         { data: [a*b    ]f32 } // 2D - Matrix
T :: struct($a, $b, $c: int)     { data: [a*b*c  ]f32 } // 3D - Tri
Q :: struct($a, $b, $c, $d: int) { data: [a*b*c*d]f32 } // 4D - Quad


/******************
* Core operations *
*******************/

// Simple matrix multiplication: AxB=O
mul :: proc(a: ^M($ar, $ac), b: ^M(ac, $bc), o: ^M(ar, bc)) {
	for r in 0..<ar {
		for c in 0..<bc {
			for k in 0..<ac {
				o.data[r*bc+c] += a.data[r*ac+k] * b.data[k*bc+c]
			}
		}
	}
}

// Matrix multiplication where the second matrix is transposed: AxBT=O
mulT :: proc(a: ^M($ar, $ac), b: ^M($br, ac), o: ^M(ar, br)) {
	for r in 0..<ar {
		for c in 0..<br {
			for k in 0..<ac {
				o.data[r*br+c] += a.data[r*ac+k] * b.data[c*ac+k]
			}
		}
	}
}

// https://ezyang.github.io/convolution-visualizer/
// Does the same as nn.Conv2d
// Does not support padding and dialation.
conv2d :: proc(img: ^T($I,$R,$C), w: ^Q($O,I,$S,S), b: ^V(O), out: ^T(O, $OR, $OC), $stride: int) {
	#assert(OR == (R-(S-1)-1)/stride+1, "Image, convolution and output width do not match")
	#assert(OC == (C-(S-1)-1)/stride+1, "Image, convolution and output height do not match")

	for o in 0..<O {
		for r in 0..<OR {
			for c in 0..<OC {
				idx := o * OR * OC + r * OC + c
				out.data[idx] += b.data[o] // Bias
				for i in 0..<I {
					for y in 0..<S {
						for x in 0..<S {
							// fmt.printfln("out[%d, %d] += img[%d, %d] * w[%d, %d]", r, c, r+y, c+x, y, x)
							out.data[idx] += img.data[i*R*C + (r*stride+y)*C + (c*stride+x)] * w.data[o*I*S*S + i*S*S + y*S + x]
						}
					}
				}
			}
		}
	}
}

add_slice :: proc(a, b, o: []f32) { for i in 0..<len(a) { o[i] = a[i] + b[i] } }
add_v :: proc(a, b, o: ^V($x))    { add_slice(a.data[:], b.data[:], o.data[:]) }
add_m :: proc(a, b, o: ^M($x,$y)) { add_slice(a.data[:], b.data[:], o.data[:]) }
add :: proc{add_slice, add_v, add_m}

relu_slice_ :: proc(o: []f32) { for i in 0..<len(o) { o[i] = (o[i] + abs(o[i])) / 2.0 } }
relu_v_ :: proc(o: ^V($x))          { relu_slice_(o.data[:]) }
relu_m_ :: proc(o: ^M($x,$y))       { relu_slice_(o.data[:]) }
relu_t_ :: proc(o: ^T($x,$y,$z))    { relu_slice_(o.data[:]) }
relu_q_ :: proc(o: ^Q($x,$y,$z,$w)) { relu_slice_(o.data[:]) }
relu_ :: proc{relu_slice_, relu_v_, relu_m_, relu_t_, relu_q_}

sigmoid_slice_ :: proc(o: []f32) { for i in 0..<len(o) { o[i] = 1.0 / (1 + linalg.exp(-o[i])) } }
sigmoid_v_ :: proc(o: ^V($x))          { sigmoid_slice_(o.data[:]) }
sigmoid_m_ :: proc(o: ^M($x,$y))       { sigmoid_slice_(o.data[:]) }
sigmoid_t_ :: proc(o: ^T($x,$y,$z))    { sigmoid_slice_(o.data[:]) }
sigmoid_q_ :: proc(o: ^Q($x,$y,$z,$w)) { sigmoid_slice_(o.data[:]) }
sigmoid_ :: proc{sigmoid_slice_, sigmoid_v_, sigmoid_m_, sigmoid_t_, sigmoid_q_}

zero_slice :: proc(o: []f32) { for i in 0..<len(o) { o[i] = 0 } }
zero_v :: proc(o: ^V($x))          { zero_slice(o.data[:]) }
zero_m :: proc(o: ^M($x,$y))       { zero_slice(o.data[:]) }
zero_t :: proc(o: ^T($x,$y,$z))    { zero_slice(o.data[:]) }
zero_q :: proc(o: ^Q($x,$y,$z,$w)) { zero_slice(o.data[:]) }
zero :: proc{zero_slice, zero_v, zero_m, zero_t, zero_q}

equal_arr :: proc(a, b: [$l]f32, eps: f32) -> bool {
	for i in 0..<l {
		if abs(a[i] - b[i]) > eps { return false }
	}
	return true
}
equal_m :: proc(a, b: ^M($A,$B),       eps: f32) -> bool { return equal_arr(a.data, b.data, eps) }
equal_t :: proc(a, b: ^T($A,$B,$C),    eps: f32) -> bool { return equal_arr(a.data, b.data, eps) }
equal_q :: proc(a, b: ^Q($A,$B,$C,$D), eps: f32) -> bool { return equal_arr(a.data, b.data, eps) }
equal :: proc{equal_arr, equal_m, equal_t, equal_q}


get_tm :: proc(i: int, t: ^T($x,$y,$z)) -> ^M(y,z) { return cast(^M(y,z))raw_data(t.data[i*y*z:]) }
get :: proc{get_tm}


/*******************
* Utiliy functions *
********************/

print_m :: proc(m: ^M($R, $C), expr := #caller_expression(m)) {
	fmt.printfln("Matrix %v (%d, %d)", expr, R, C)
	for r in 0..<R {
		for c in 0..<C {
			d := m.data[r*C+c]
			fmt.printf("%s%.4f ", "" if d < 0 else " ", d)
		}
		fmt.println()
	}
}

print_t :: proc(t: ^T($A,$B,$C), expr := #caller_expression(t)) {
	fmt.printfln("Tri-Tensor %v (%d, %d, %d)", expr, A,B,C)
	idx := 0
	for a in 0..<A {
		for b in 0..<B {
			for c in 0..<C {
				d := t.data[idx]
				idx += 1
				fmt.printf("%s%.4f ", "" if d < 0 else " ", d)
			}
			fmt.println()
		}
		fmt.println()
	}
}

print :: proc{print_m, print_t}

imshow :: proc(m: ^M($R, $C), expr := #caller_expression(m)) {
	fmt.printfln("Matrix %v (%d, %d)", expr, R, C)
	for r in 0..<R {
		for c in 0..<C {
			d := m.data[r*C+c]
			fmt.print("." if d < 0.5 else "#")
		}
		fmt.println()
	}
}

as_vs :: proc($a:          int, data: []f32) -> ^V(a)       { return cast(^V(a))raw_data(data) }
as_ms :: proc($a,$b:       int, data: []f32) -> ^M(a,b)     { return cast(^M(a,b))raw_data(data) }
as_ts :: proc($a,$b,$c:    int, data: []f32) -> ^T(a,b,c)   { return cast(^T(a,b,c))raw_data(data) }
as_qs :: proc($a,$b,$c,$d: int, data: []f32) -> ^Q(a,b,c,d) { return cast(^Q(a,b,c,d))raw_data(data) }
as :: proc{as_vs, as_ms, as_ts, as_qs}
