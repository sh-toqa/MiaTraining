# Task 12.5 Clean Coder

## **Problem Breakdown**

### **1. Rotary Encoder Specifications**
* **Encoder Resolution:** 7 pulses per revolution (PPR).
* **Gearbox Ratio:** 30:1.
* **Wheel Diameter:** 20cm.
* **Maximum Robot Speed:** 1m/s.
* **Noisy Signal:** The encoder output is noisy, requiring signal conditioning via a low-pass filter.

### **2. Target**

* Calculate the ***cutoff frequency*** for a first-order low-pass filter (LPF) to reduce noise while retaining useful signal information.


## **Analysis**
### **Step 1 :   Calculate the Encoder's Effective PPR**

- A rotary encoder measures the rotation of the motor shaft. It generates a certain number of electrical pulses per revolution of the shaft. In this case, the encoder *generates 7 pulses per revolution (PPR) of the motor shaft*.
- Now, the motor is attached to a 30:1 gearbox. This means that the motor shaft *spins 30 times for every full revolution of the wheel*. 
> Therefore, for every revolution of the wheel, the motor turns 30 times, and each of those turns produces 7 pulses from the encoder.
$$Effective PPR = Encoder PPR × Gearbox ratio = 7×30=210 PPR $$

### **Step 2 : Calculate Wheel RPM at Maximum Speed**

> The robot can move at a maximum speed of 1 meter per second (m/s). The wheel's diameter is 20 cm. 

To understand how fast the wheel is rotating, we need to know the **circumference of the wheel**, which is the distance it covers in one full revolution.
$$ 
Circumference=π×Diameter=π×0.2m=0.6283m
$$
> This means that every time the wheel rotates once, the robot moves forward 0.6283 meters.

Now, if the robot is moving at 1 m/s, we can calculate how many **revolutions per second (RPS)** the wheel is making:

$$
\text{RPS} = \frac{\text{Robot speed}}{\text{Circumference of the wheel}} = \frac{1}{0.6283} = 1.591 \, \text{RPS}
$$
*To convert RPS to RPM (revolutions per minute), multiply by 60:*
$$ 
RPM=1.591×60=95.45RPM
$$
> So, at the maximum speed of 1 m/s, the wheel is rotating at about 95.45 RPM.

### **Step 3 : Calculate Encoder Frequency**
> We now know that the wheel spins at about **95.45 RPM at maximum speed**, and that the encoder produces **210 pulses per revolution** of the wheel.

The encoder generates pulses at a rate determined by the wheel’s rotational speed (in RPM) and the effective PPR.
The **pulse frequency (number of pulses per second)** is then:
$$
f_{\text{encoder}} = \text{Effective PPR} \times \text{RPS} = 210 \times 1.591 = 334.11 \, \text{Hz}
$$
> This means that the encoder is generating a signal at a frequency of 334 Hz at the robot's maximum speed. This is the frequency of the useful signal that tells us the wheel's speed.

### **Step 4 : Dealing with Noise Using a Low-Pass Filter (LPF)**
The key parameter in an LPF is the **cutoff frequency** .

*Frequencies below the cutoff are passed, while frequencies above it are attenuated.*
* The encoder's useful signal is at about 334 Hz.
* To ensure that the filter captures this signal, we typically set the cutoff frequency a bit higher than the signal's highest frequency.
* Typically, we choose the cutoff frequency to be **around 1.5 to 2 times the highest signal frequency**.
$$
f_c = 1.5 \times f_{\text{encoder}} = 1.5 \times 334.11 = 501.17 \, \text{Hz}
$$
> Thus, a suitable cutoff frequency for the low-pass filter is around **500 Hz**.

### **Step 5 : Low-Pass Filter Implementation**
A first-order low-pass filter in continuous-time has the following transfer function:

$$
H(s) = \frac{\omega_c}{s+\omega_c}
$$
Where :
* $w_c$ is the cutoff angular frequency in radians per second:
$$
\omega_c = 2 \pi f_c = 2 \pi \times 500 = 3141.59 \, \text{rad/s}
$$
* $s$ is the Laplace variable (complex frequency).
