# Short Term Person Recognition using a body-type classifier and Expectation Maximization to recognize clothing
The goal of this project is to take a bare-bones mobile robot running ROS and provide a full software pipeline to make this robot into an autonomous agent, capable of assisting humans in a variety of tasks.

## Algorithm

P(Individual | Data) =  a * P(Individual | Clothing) + b * P(Individual | Body Type)

P(Individual | Clothing) = {

Steps: (0 - 2 are E&M in Color Space)
		0: Generate a random horizontal line (theta), classify points below and above it.
		1: Build color histograms, Histogram1 (H1) is above the theta line, and Histogram 2 (H2) is below. One histogram will likely contain data that isn’t like the rest of its data and is more like the other histograms data, so the line must be moved.
		2: Reclassify Points, P(p | H1), P(p | H2) then relabel points
		3: (Physical Space) K-Means step on theta line, mean of the 2 distributions to converge to middle points (probably knees and ribs)
		4: repeat steps 1 to 3 until convergence

		}

P(Individual | Body Type) = {
-Anthropometric Distances for shape-based biometric human identification-Create a skeleton of points and distance vectors between body parts using input images of each person. Create point vectors that clearly show visible inter-joint distances and the euclidean distance should be invariant of most poses.
-Superimpose these points on to a new image.

}

##  Clothing Expectation Maximization On Live Data
One of the most important steps in classifying humans within our framework is clothing recognition. In a typical office, humans check in with security or do some sort of sign-in at the beginning of the day. We can add to this sign-in process by having a camera take pictures of the approaching worker. In this model, those pictures can be used to train an intelligent agent on what kind of clothing it should expect each individual to be wearing. Once each individual's appearance is documented, the intelligent robot can then recognize and provide assistance to specific humans.

###  Random Line
The robot will need to recognize people in real time and the first step of this process is to generate a random line on a live image or image stream. The purpose of this is to define a boundary where the colors in an image to differ. This could define the boundary between a person's pants and shirt or the outfit is monochromatic, it will define the boundary between the person's outfit and skin.

![A-Probabilistic-Framework-for-Short-Term-Person-Recognition-line](https://raw.githubusercontent.com/julianweisbord/A-Probabilistic-Framework-for-Short-Term-Person-Recognition/master/data_capture/manipulated_images/line.jpeg)
###  Color Histograms
Next we can create color histograms, to acquire info about what colors are most present in the live image data.
![A-Probabilistic-Framework-for-Short-Term-Person-Recognition-color_histograms](https://raw.githubusercontent.com/julianweisbord/A-Probabilistic-Framework-for-Short-Term-Person-Recognition/master/data_capture/manipulated_images/color_histograms.png)
##  Body Classification

##  Dependencies
pip install -r dependencies.txt
