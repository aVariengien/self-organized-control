



const output_cells = [[13,16], [17,16]];
const input_cells = [ [11, 26],[25,20],[5,20],[19,6],[19,26],[5,12],[25,12] ,[11, 6]];

function init_grid(){

	var grid = tf.randomUniform( [1, 32,32,6], -0.1, 0.1);  

	var grid_buffer = tf.buffer(grid.shape, grid.dtype, grid.dataSync());

	for (let x=0; x<32; x++) {
		for (let y=0; y<32; y++){
			grid_buffer.set(0., 0, x,y, 2);
			grid_buffer.set(0., 0, x,y, 3);
		};
	};
	for (let j=0; j<output_cells.length ; j++) {
		grid_buffer.set(1., 0, output_cells[j][0], output_cells[j][1], 3);
		grid_buffer.set(1., 0, output_cells[j][0], output_cells[j][1], 4);
		grid_buffer.set(1., 0, output_cells[j][0], output_cells[j][1], 5);
	};

	for (let j=0; j<input_cells.length ; j++) {
		
		grid_buffer.set(1., 0, input_cells[j][0], input_cells[j][1], 1);
		grid_buffer.set(1., 0, input_cells[j][0], input_cells[j][1], 2);
		grid_buffer.set(1., 0, input_cells[j][0], input_cells[j][1], 4);
		grid_buffer.set(1., 0, input_cells[j][0], input_cells[j][1], 5);
	};

	return grid_buffer.toTensor();
};



//create mask to speed up computation 
//neg mask
var cut_inp = tf.ones( [1,32,32,6]);
var cut_inp_buf = tf.buffer(cut_inp.shape, cut_inp.dtype, cut_inp.dataSync());
for (let j=0; j<input_cells.length ; j++) {
	cut_inp_buf.set(0., 0, input_cells[j][0], input_cells[j][1], 0);
	cut_inp_buf.set(0., 0, input_cells[j][0], input_cells[j][1], 1);
	cut_inp_buf.set(0., 0, input_cells[j][0], input_cells[j][1], 4);
	cut_inp_buf.set(0., 0, input_cells[j][0], input_cells[j][1], 5);
};
cut_inp = cut_inp_buf.toTensor();

//positive mask
var add_values = tf.zeros( [1,32,32,6]);
var add_values_buf = tf.buffer(add_values.shape, add_values.dtype, add_values.dataSync());

for (let j=0; j<output_cells.length ; j++) {
	add_values_buf.set(1., 0, output_cells[j][0], output_cells[j][1], 4);
	add_values_buf.set(1., 0, output_cells[j][0], output_cells[j][1], 5);
};

for (let j=0; j<input_cells.length ; j++) {
	add_values_buf.set(1., 0, input_cells[j][0], input_cells[j][1], 1);
	add_values_buf.set(1., 0, input_cells[j][0], input_cells[j][1], 4);
	add_values_buf.set(1., 0, input_cells[j][0], input_cells[j][1], 5);
};

//cut io chan for noisy update
var cut_io_chan = tf.ones( [1,32,32,6]);
var cut_io_chan_buf = tf.buffer(cut_io_chan.shape, cut_io_chan.dtype, cut_io_chan.dataSync());
for (let x=0; x<32 ; x++) {
	for (y=0; y<32; y++) {
		cut_io_chan_buf.set(0., 0, x, y, 2);
		cut_io_chan_buf.set(0., 0, x, y, 3);
	};
};

cut_io_chan = cut_io_chan_buf.toTensor();
 
 


function runCA(gr,values, nb_step) {

	var g = gr.clone();
	gr.dispose();
	
	for (let j=0; j<values.length ; j++) {
		add_values_buf.set(values[j], 0, input_cells[j][0], input_cells[j][1], 0);
	};
	var add_values_tensor = add_values_buf.toTensor();

	for (let i=0; i<nb_step; i++)
	{
		g = tf.tidy( () => {
			var dx = CAmodel.predict(g);
			if (add_noise){
				var noise = tf.randomUniform([1, 32,32,6], -0.01, 0.01);
				dx = tf.add(dx, noise);
				dx = tf.mul(dx, cut_io_chan);
			};
			
			var uniform_update_mask = tf.randomUniform([1, 32,32,1], 0., 1.);
			uniform_update_mask = tf.less( uniform_update_mask, 0.5);
			
			uniform_update_mask = tf.cast(uniform_update_mask, 'float32');
			dx = tf.mul(dx, uniform_update_mask);
			return tf.add(g, dx);});
		
		g = tf.tidy( () => {
			return tf.add(tf.mul(g,cut_inp), add_values_tensor);});
	};

	return g
};

//function to display the grids

var RED_TENSOR = tf.tensor3d([255, 0, 0], [1, 1, 3]);
var BLUE_TENSOR = tf.tensor3d([0, 0, 255], [1, 1, 3]);
var WHITE_TENSOR = tf.tensor3d([255, 255, 255], [1, 1, 3]);

function apply_color_scale(info_chan){
	var blue_side_mask = tf.cast(tf.less(info_chan,0), 'float32');
	var red_side_mask = tf.sub(1.,blue_side_mask);
	var abs_val = tf.abs(info_chan);
	var w_val = tf.sub(1., abs_val);
	var BLUE=  tf.mul(tf.mul(blue_side_mask, abs_val), BLUE_TENSOR);
	var RED =  tf.mul(tf.mul(red_side_mask, abs_val), RED_TENSOR);
	var WHITE = tf.mul(w_val, WHITE_TENSOR);
	var final_val = tf.add(tf.add(BLUE, RED),WHITE);
	return final_val;
};

function get_chans_tensors(gri) {
	
	var info_chan = tf.tidy( () => {
		
		var a = tf.stridedSlice(gri, [0,0,0,0], [1,32,32,1]);
		a = tf.reshape(a, [32,32,1]);
		a = tf.clipByValue( a, -1, 1);
		return apply_color_scale(a);
	});
	
	
	var hid_chans = tf.tidy( () => {
		
		var hid_chan1 = tf.stridedSlice(gri, [0,0,0,1], [1,32,32,2]).reshape([32,32,1]);
		
		var hid_chan23 = tf.stridedSlice(gri, [0,0,0,4], [1,32,32,6]).reshape([32,32,2]);

		var h = tf.concat([hid_chan1, hid_chan23], -1);
		h = tf.clipByValue( h, -1, 1);
		return tf.mul(tf.mul(h, 0.5).add(0.5),255);
	});
	
	return [info_chan, hid_chans];
	
};

function disp_grid(grid) {
	const [info_chan, hid_chans] = get_chans_tensors(grid);
	var info_chan_buf = tf.buffer(info_chan.shape, info_chan.dtype, info_chan.dataSync());
	var hid_chan_buf = tf.buffer(hid_chans.shape, hid_chans.dtype, hid_chans.dataSync());

	
    buffer_inf = new Uint8ClampedArray(32 * 32 * 4);
    buffer_hid = new Uint8ClampedArray(32 * 32 * 4);
	
	for(var x = 0; x < 32; x++) {
		for(var y = 0; y < 32; y++) {
			var pos = (x * 32 + y) * 4; // position in buffer based on x and y
			buffer_inf[pos] = info_chan_buf.get(x,y,0);           // some R value [0, 255]
			buffer_inf[pos+1] = info_chan_buf.get(x,y,1);           // some G value
			buffer_inf[pos+2] = info_chan_buf.get(x,y,2);           // some B value
			buffer_inf[pos+3] = 255;           // set alpha channel
			
			buffer_hid[pos] = hid_chan_buf.get(x,y,0);           // some R value [0, 255]
			buffer_hid[pos+1] = hid_chan_buf.get(x,y,1);           // some G value
			buffer_hid[pos+2] = hid_chan_buf.get(x,y,2);           // some B value
			buffer_hid[pos+3] = 255;           // set alpha channel
		}
	}
	
	canvas_data_hid.width = 32;
	canvas_data_hid.height = 32;
	
	canvas_data_info.width = 32;
	canvas_data_info.height = 32;
	
	var idata1 = ctx_hid_data.createImageData(32, 32);
	var idata2 = ctx_inf_data.createImageData(32, 32);

	// set our buffer as source
	idata1.data.set(buffer_hid);
	idata2.data.set(buffer_inf);
	// update canvas with new data
	ctx_hid_data.putImageData(idata1, 0, 0);
	ctx_inf_data.putImageData(idata2, 0, 0);
	
	ctx_info.clearRect(0, 0, DISPLAY_H, DISPLAY_H);
	ctx_info.drawImage(canvas_data_info, 0, 0, DISPLAY_H, DISPLAY_H);
							
	ctx_hid.clearRect(0, 0, DISPLAY_H, DISPLAY_H);
	ctx_hid.drawImage(canvas_data_hid, 0, 0, DISPLAY_H, DISPLAY_H);
	
	info_chan.dispose();
	hid_chans.dispose();
};


//initialisation

var grid = tf.tidy( () => {return init_grid()});

var add_noise = false;
var onPause = false;

let env = new CartPoleEnv();

var values = [0,0,0,0,0,0,0,0];

var nb_step_before_act = 1;
var in_factors = [2., 0.25, 4., 0.15];


var prec = Date.now();
var t1 = Date.now();

var model_to_load= true;
var current_model = '1';
var model_path = 'demo/models/';

function LoadModel(s) {
	return tf.loadLayersModel(model_path+'model'+s+'/model.json');
};

function ChangeModel(model_nb){
	current_model = model_nb;
	model_to_load = true;
	grid = tf.tidy( () => {return init_grid()});
};

function UpdateNoise() {
	var checkBox = document.getElementById("add_noise");
	add_noise = checkBox.checked ;
};

function ShowCaption() {
	var checkBox = document.getElementById("show_caption");
	DrawCap = checkBox.checked ;
};


function ResetGrid() {
	 grid = tf.tidy( () => {return init_grid()});
 };



function PlayPause() {
	var playPauseImg = document.getElementById("playPauseImg");
	if (onPause) {
		onPause = false;
		playPauseImg.src = "img/pause.png";
	}
	else {
		onPause = true;
		playPauseImg.src = "img/play.png";
	};
};

function UpdateSpeed(val) {
	if (val == 1)
	{	
		msInterval = 500;
		ca_step_per_update = 1;
	};
	if (val == 2)
	{	
		msInterval = 80;
		ca_step_per_update = 1;
	};
	if (val == 3)
	{	
		msInterval = 20;
		ca_step_per_update = 1;
	};
	if (val == 4)
	{	
		msInterval = 100;
		ca_step_per_update = 10;
	};
	if (val == 5)
	{	
		msInterval = 10;
		ca_step_per_update = 60;
	};
};

function roundDecimal(nombre, precision){
    var tmp = Math.pow(10, precision);
    return Math.round( nombre*tmp )/tmp;
}



 function getCursorPosition(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
	return [x,y];
    
}

var DrawCap = true;
function drawCaption(c) {
	if (DrawCap) {
		var s = DISPLAY_H/32;
		for (let j=0; j<input_cells.length; j++) {
			drawCircle(c,input_cells[j][0]*s, input_cells[j][1]*s,s*0.25, "black", 0.5*s);
		};
		
		drawCircle(c,output_cells[0][0]*s, output_cells[0][1]*s,s*0.25, "DarkGreen", 0.5*s);
		drawCircle(c,output_cells[1][0]*s, output_cells[1][1]*s,s*0.25, "orange", 0.5*s);
		
		c.fillStyle = "DarkGreen";
		c.font = "bold 20px Roboto";
		
		c.fillText("Left", output_cells[0][0]*s+15, output_cells[0][1]*s-40); 
		
		c.fillStyle = "orange";
		c.fillText("Right", output_cells[1][0]*s-25, output_cells[1][1]*s+40); 
		
		
		c.font = "13px Roboto";
		

		c.fillStyle = "black";
		c.fillText("Cart position", input_cells[0][1]*s-20, input_cells[0][0]*s-8); 
		c.fillText("Cart position", input_cells[1][1]*s-10, input_cells[1][0]*s+25); 
		
		c.fillText("Cart velocity", input_cells[2][1]*s-30, input_cells[2][0]*s-8); 
		c.fillText("Cart velocity", input_cells[3][1]*s-50, input_cells[3][0]*s+25); 
		
		c.fillText("Pole angle", input_cells[4][1]*s-10, input_cells[4][0]*s+25); 
		c.fillText("Pole angle", input_cells[5][1]*s-30, input_cells[5][0]*s-8);
		
		
		c.fillText("Pole angular", input_cells[6][1]*s-40, input_cells[6][0]*s+25); 
		c.fillText("velocity", input_cells[6][1]*s-40, input_cells[6][0]*s+38); 
		
		c.fillText("Pole angular", input_cells[7][1]*s-40, input_cells[7][0]*s-28); 
		c.fillText("velocity", input_cells[7][1]*s-40, input_cells[7][0]*s-8); 
		
	};
	
};

//create damage
function drawCircle(c, x,y,R, col, offset=0) {
	c.beginPath();
	c.arc(y+offset, x+offset, R, 0, 2 * Math.PI);
	c.fillStyle = col;
	c.fill(); 
};
var [xd, yd] = [0,0];
var radiusDam = 0;

var inDamage=false;

function damageGrid(event, canvas) {
	if (inDamage) {
		[yd,xd] = getCursorPosition(canvas, event);
		var R = 5;
		
		var xg = xd*32/DISPLAY_H;
		var yg = yd*32/DISPLAY_H;
		
		grid = tf.tidy( () => {
			var grid_buffer = tf.buffer(grid.shape, grid.dtype, grid.dataSync());
			for (let x=0; x <32; x++) {
				for (let y=0; y<32; y++) {
					if ( (x-xg)*(x-xg) + (y-yg)*(y-yg) < R*R)
					{
						grid_buffer.set(randUnif(-1, 1), 0,x,y,0);
						grid_buffer.set(randUnif(-1, 1), 0,x,y,1);
						grid_buffer.set(randUnif(-1, 1), 0,x,y,4);
						grid_buffer.set(randUnif(-1, 1), 0,x,y,5);
					}
					
				}
			}
			return grid_buffer.toTensor();
		});
		radiusDam = R;
		setTimeout(showDamageCircle, 20);
	}
};


async function showDamageCircle() {
	
	if (radiusDam>1) {
		if (DrawCap) {
			disp_grid(grid);
			drawCircle(ctx_hid, xd, yd, radiusDam*DISPLAY_H/32, "rgba(255, 0, 0, 0.6)");
			drawCircle(ctx_info, xd, yd, radiusDam*DISPLAY_H/32, "rgba(255, 0, 0, 0.6)");
			drawCaption(ctx_hid);
			drawCaption(ctx_info);
		};
	}
	else {
		if (DrawCap) {
			disp_grid(grid);
			drawCaption(ctx_hid);
			drawCaption(ctx_info);
		};
	};

	radiusDam -=1;
	console.log("rad:",radiusDam);
	if (radiusDam>0) {
		setTimeout(showDamageCircle, 20);
	};
};


function beginDamage(event, canvas) {
	inDamage = true;
	damageGrid(event, canvas);
};
function endDamage() {
	inDamage = false;
};



var showPush = false;
function pertrubCartpole(event, canvas) {
	showPush = true;
	[x,y] = getCursorPosition(canvas, event);

	if (x > (obs[0]+2.4)*env.sc) {
		env.step(-1);
		var image = document.getElementById("push_left");
		ctx_cartpole.drawImage(image, x-25, y-25, 50, 50);
		
	}
	else
	{
		env.step(2);
		var image = document.getElementById("push_right");
		ctx_cartpole.drawImage(image, x-25, y-25, 50, 50);
	};
	setTimeout( () => {env.render(ctx_cartpole); showPush=false;}, 100);
};

var msInterval = 20;
var ca_step_per_update = 1;
var obs;
async function update () { 
							

							if (model_to_load){
								CAmodel = await LoadModel(current_model);
								model_to_load = false;
							};
							
							if (!onPause) {
								t1 = Date.now();
								var ca_step_per_sec = roundDecimal((1000*ca_step_per_update/(t1-prec)), 0);
								var cartpole_step_per_sec = roundDecimal((1000*ca_step_per_update/(55*(t1-prec))), 2);
								document.getElementById("speed_label").innerHTML = "("+ca_step_per_sec+" CA step/s - "+cartpole_step_per_sec+" cart-pole step/s)";
								prec = t1;
								
								
								disp_grid(grid);
								if (radiusDam>1) {
									if (DrawCap) {
										drawCircle(ctx_hid, xd, yd, radiusDam*DISPLAY_H/32, "rgba(255, 0, 0, 0.6)");
										drawCircle(ctx_info, xd, yd, radiusDam*DISPLAY_H/32, "rgba(255, 0, 0, 0.6)");
									};
									//radiusDam -= 3*ca_step_per_update;
								};
								
								drawCaption(ctx_hid);
								drawCaption(ctx_info);

								if (nb_step_before_act==0) {
									var [left, right] = tf.tidy( () => {
										var new_grid_buffer = tf.buffer(grid.shape, grid.dtype, grid.dataSync());
										var l = new_grid_buffer.get(0, output_cells[0][0], output_cells[0][1], 0);
										var r = new_grid_buffer.get(0, output_cells[1][0], output_cells[1][1], 0);
										return [l,r];
									});
									
									nb_step_before_act = Math.floor(randUnif(50,60));
									
									var action = 1;
									if (left > right) {
										action = 0;
									};
									
									obs = env.step(action);
									
									for (let k=0; k<8; k++) {
										values[k] = obs[Math.floor(k/2)]*in_factors[Math.floor(k/2)];
									};
									
									if (!showPush)
									{
										env.render(ctx_cartpole);
									};
								};
								
								if (nb_step_before_act>ca_step_per_update)
								{
									grid = tf.tidy(
										
										() => {
											return runCA(grid, values, ca_step_per_update);
									});
									nb_step_before_act -= ca_step_per_update;
								}
								else
								{
									grid = tf.tidy(
										
										() => {
											return runCA(grid, values, nb_step_before_act);
									});
									nb_step_before_act = 0;
								};
							}
							console.log("yo");
							setTimeout(update, msInterval);

						    };
var canvas_data_info;
var canvas_data_hid;
var canvasbig_hid;
var canvasbig_info;
var ctx_hid;
var ctx_info;
var canvas_cartpole;
var ctx_cartpole;
var canvas_plots;
var ctx_plots;

async function init() {
	CAmodel = await LoadModel(current_model);
	canvas_data_info = document.getElementById("ca_grid_info");
	canvas_data_hid = document.getElementById("ca_grid_hid");
	canvasbig_hid = document.getElementById("display_hid");
	canvasbig_info= document.getElementById("display_info");

	ctx_hid = canvasbig_hid.getContext('2d');
	ctx_hid.imageSmoothingEnabled= false;
	ctx_info = canvasbig_info.getContext('2d');
	ctx_info.imageSmoothingEnabled= false;
	
	canvas_cartpole = document.getElementById("cartpole");
	ctx_cartpole = canvas_cartpole.getContext('2d');
	
	canvas_plots = document.getElementById("plots");
	ctx_plots = canvas_cartpole.getContext('2d');
	
	ctx_hid_data = canvas_data_hid.getContext('2d');
	ctx_inf_data = canvas_data_info.getContext('2d');
	
	
	document.getElementById("show_caption").checked = true;
	document.getElementById("add_noise").checked = false;
	document.getElementById("speed_slider").value = "3";
	setTimeout(update, msInterval);
	preLoad();
	
};


init();
window.onload = init;


