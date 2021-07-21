
function randUnif(low, high) {
	return Math.random() * (high-low) + low;
};

class CartPoleEnv {
  constructor() {

        this.gravity = 9.8;
        this.masscart = 1.0;
        this.masspole = 0.1;
        this.total_mass = (this.masspole + this.masscart);
        this.length = 0.5  ;
        this.polemass_length = (this.masspole * this.length);
        this.force_mag = 10.0;
        this.tau = 0.02  ;
        this.theta_threshold_radians = 30 * 2 * Math.PI / 360;
        this.x_threshold = 2.4;
		this.step_nb = 0;
		
		this.sc = DISPLAY_H/3;
		
		this.reset();
  }
  
  reset() {
	  this.step_nb = 0;
	  this.state = [0,0,0,0];
	  
	  for (let i=0; i<this.state.length; i++) {
		  this.state[i] = randUnif(-0.05, 0.05);
	  }; 
	  return this.state;
  }
  
  set_state(s) {
	  this.state = s;
  }
  
  step(action) {
	  
		//console.log(this.step_nb);
		this.step_nb +=1;
		var [x, x_dot, theta, theta_dot] = this.state;
		var force = (action-0.5)*this.force_mag*2;
		
		var costheta = Math.cos(theta);
		var sintheta = Math.sin(theta);
		
		var temp = (force + this.polemass_length * Math.pow(theta_dot, 2) * sintheta) / this.total_mass;
		var thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0 / 3.0 - this.masspole * Math.pow(costheta, 2) / this.total_mass));
		var xacc = temp - this.polemass_length * thetaacc * costheta / this.total_mass

		x = x + this.tau * x_dot;
        x_dot = x_dot + this.tau * xacc;
        theta = theta + this.tau * theta_dot;
        theta_dot = theta_dot + this.tau * thetaacc;
		
		this.state = [x, x_dot, theta, theta_dot];
		
		var done = (x < -this.x_threshold || x > this.x_threshold || theta < -this.theta_threshold_radians || theta > this.theta_threshold_radians);
		
		if (done){
			console.log("RESET"+x+" "+theta+" "+this.x_threshold+ " "+ this.theta_threshold_radians+" "+done);
			
			this.reset();
		};
		return this.state;
  }
  
  render(c) {
		var polelen = this.sc*2*this.length;
		var b = this.sc*this.length*6;
		var t = 0;
		var l = 0;
		var r = this.sc*this.x_threshold*2;
		var [x, x_dot, theta, theta_dot] = this.state;
		c.setTransform(1, 0, 0, 1, 0, 0);
		
		c.clearRect(0, 0, r, b);
		
		c.fillStyle = "black";
		c.font = "20px Roboto";
		var s = "Step "+this.step_nb
		c.fillText(s, 30, 50); 
		
		//draw frame
		c.beginPath();
		c.moveTo(l, t);
		c.lineTo(r, t);
		c.lineTo(r, b);
		c.lineTo(l, b);
		c.lineTo(l, t);
		c.stroke(); 

		var tr_y = b-this.sc*this.length*0.5;
		var cartH = this.sc*this.length*0.48;
		var cartW = this.sc*this.length*0.8;
		var poleW = this.sc*this.length*0.16;

		// draw track
		c.beginPath();
		c.moveTo(l, tr_y);
		c.lineTo(r, tr_y);
		c.stroke(); 

		//draw cart
		c.setTransform(1, 0, 0, 1, 0, 0);
		
		c.translate(r/2+x*this.sc, tr_y);
		
		c.beginPath();
		var [b,t,l,r] = [-cartH/2, cartH/2, -cartW/2, cartW/2];
		
		c.moveTo(l, t);
		c.lineTo(r, t);
		c.lineTo(r, b);
		c.lineTo(l, b);
		c.lineTo(l, t);
		c.fillStyle = "black";
		c.fill();

		c.setTransform(1, 0, 0, 1, 0, 0);
		
		
		
		// draw pole
		r = this.sc*this.x_threshold*2;
		c.translate(r/2+x*this.sc, tr_y);
		c.rotate(theta);
		c.translate(0, -polelen*0.5);
		
		c.beginPath();
		[b,t,l,r] = [-polelen/2, polelen/2, -poleW/2, poleW/2];
		
		c.moveTo(l, t);
		c.lineTo(r, t);
		c.lineTo(r, b);
		c.lineTo(l, b);
		c.lineTo(l, t);
		c.fillStyle = "rgba(204, 153, 102, 255)";
		c.fill();
		c.setTransform(1, 0, 0, 1, 0, 0);


		r = this.sc*this.x_threshold*2;
		c.translate(r/2+x*this.sc, tr_y);
		

		// draw axle
		c.beginPath();
		c.arc(0, 0, poleW/2, 0, 2 * Math.PI);
		c.fillStyle = "rgba(128, 128, 204, 255)";
		c.fill(); 
		c.setTransform(1, 0, 0, 1, 0, 0);
		
		
  }
};