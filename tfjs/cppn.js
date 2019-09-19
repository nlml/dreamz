var canvas = document.getElementById("myCanvas");
var intervalHandle = null;
var sizeMulti = 4;

function getXYMesh(size, r = 3.0 ** 0.5) {
	if (size[0] >= size[1]) {
		ratio = size[0] / size[1]
		rng0 = tf.linspace(-r * ratio, r * ratio, size[0])
		rng1 = tf.linspace(-r, r, size[1])
	} else {
		ratio = size[1] / size[0]
		rng0 = tf.linspace(-r, r, size[0])
		rng1 = tf.linspace(-r * ratio, r * ratio, size[1])
	}
	return tf.stack([
		rng0.reshape([-1, 1]).tile([1, size[1]]),
		rng1.reshape([1, -1]).tile([size[0], 1])
	], 0).reshape([1, 2, size[0], size[1]]);
}

class CompositeActivation extends tf.layers.Layer {
	constructor() {
		super({});
		this.supportsMasking = true;
	}
	computeOutputShape(inputShape) {
		return [inputShape[0], 2 * inputShape[1], inputShape[2], inputShape[3]]
	}
	call(inputs, kwargs) {
		let x = inputs;
		if (Array.isArray(x)) {
			x = x[0];
		}
		this.invokeCallHook(inputs, kwargs);
		const atand = tf.atan(x)
		const left = atand.mul(1.0 / 0.67)
		const right = atand.mul(atand).mul(1.0 / 0.6)
		return tf.concat([left, right], 1);
	}
	static get className() {
		return 'CompositeActivation';
	}
}
tf.serialization.registerClass(CompositeActivation); // Needed for serialization.


function compositeActivation() {
	return new CompositeActivation();
}

function getModelOld(inputSize) {

	const model = tf.sequential();
	model.add(tf.layers.conv2d({
		inputShape: [4, inputSize[0], inputSize[1]],
		kernelSize: 1,
		filters: 8,
		strides: 1,
		kernelInitializer: 'glorotNormal',
		dataFormat: 'channelsFirst'
	}));
	model.add(compositeActivation());
	for (var i = 0; i < 14; i++) {
		model.add(tf.layers.conv2d({
			kernelSize: 1,
			filters: 6,
			strides: 1,
			kernelInitializer: 'glorotNormal',
			dataFormat: 'channelsFirst'
		}));
		model.add(compositeActivation());
	}
	model.add(tf.layers.conv2d({
		kernelSize: 1,
		filters: 3,
		strides: 1,
		kernelInitializer: 'glorotNormal',
		dataFormat: 'channelsFirst',
		activation: 'sigmoid'
	}));
	return model
}

function getModel(inputSize) {

	const model = tf.sequential();
	model.add(tf.layers.conv2d({
		inputShape: [4, inputSize[0], inputSize[1]],
		kernelSize: 1,
		filters: 8,
		strides: 1,
		kernelInitializer: 'glorotNormal',
		dataFormat: 'channelsFirst'
	}));
	model.add(compositeActivation());
	for (var i = 0; i < 5; i++) {
		model.add(tf.layers.conv2d({
			kernelSize: 1,
			filters: 8,
			strides: 1,
			kernelInitializer: 'glorotNormal',
			dataFormat: 'channelsFirst'
		}));
		model.add(compositeActivation());
	}
	model.add(tf.layers.reshape({
		targetShape: [8, inputSize[0] * sizeMulti, inputSize[1] * sizeMulti]
	}));
	model.add(tf.layers.conv2d({
		kernelSize: 1,
		filters: 3,
		strides: 1,
		kernelInitializer: 'glorotNormal',
		dataFormat: 'channelsFirst',
		activation: 'sigmoid'
	}));
	return model
}

var canvasMouseDistFromCtrX = 0;
var canvasMouseDistFromCtrY = 0;

function getMousePos(canvas, evt) {
	var rect = canvas.getBoundingClientRect();
	canvasMouseDistFromCtrX = (evt.clientX - rect.left) / canvas.clientWidth - 0.5;
	canvasMouseDistFromCtrY = (evt.clientY - rect.top) / canvas.clientHeight - 0.5;
}

function mousey(evt) {
	getMousePos(canvas, evt);
	// console.log(canvasMouseDistFromCtrX, canvasMouseDistFromCtrY);
}

function reRender() {
	var canvas = document.getElementById("myCanvas");
	canvas.width = 100;
	canvas.height = 100;
	const size = [canvas.height, canvas.width];
	const model = getModel(size);
	var xy = getXYMesh(size);
	var mouseMesh = getXYMesh(size, r=1.0);

	var ctx2 = canvas.getContext("2d");

	var c1 = document.createElement("canvas");
	c1.width = size[0];
	c1.height = size[1];
	var ctx1 = c1.getContext("2d");

	var frameIdx = 0;
	if (intervalHandle !== null) {
		clearInterval(intervalHandle);
		intervalHandle = null;
	}
	intervalHandle = setInterval(function() {
		// console.log(frameIdx);
		// console.log(canvasMouseDistFromCtrX);
		const tt = tf.tensor([canvasMouseDistFromCtrY, canvasMouseDistFromCtrX]).reshape([1, 2, 1, 1])
		const ss = tf.square(mouseMesh.sub(tt));
		var output = model.apply(xy.add(tt).concat(ss, 1))
		output = output.concat(
			tf.ones([1, 1, size[0] * sizeMulti, size[1] * sizeMulti]), 1).mul(
			255).toInt().transpose([0, 2, 3, 1]);
		const values = output.dataSync();

		var imgData = ctx1.createImageData(size[0] * sizeMulti, size[1] * sizeMulti);

		imgData.data.set(Uint8ClampedArray.from(values));
		ctx1.putImageData(imgData, 0, 0);

		ctx2.mozImageSmoothingEnabled = false;
		ctx2.webkitImageSmoothingEnabled = false;
		ctx2.msImageSmoothingEnabled = false;
		ctx2.imageSmoothingEnabled = false;
		ctx2.drawImage(c1, 0, 0, size[0], size[1]);
		frameIdx++;
	}, 50);
}
canvas.addEventListener('click', reRender, false);
window.addEventListener('resize', reRender, false);