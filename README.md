# Short Term Person Recognition using a body-type classifier and Expectation Maximization to recognize clothing
The goal of this project is to take a barebones mobile robot running ROS and provide a full software pipeline to make this robot into an autonomous agent, capable of assisting humans in a variety of tasks.


##  Clothing Expectation Maximization On Live Data
One of the most important steps in classifying humans within our framework is clothing recognition. In a typical office, humans check in with security or do some sort of sign-in at the beginning of the day. We can add to this sign-in process by having a camera take pictures of the approaching worker. In this model, those pictures can be used to train an intelligent agent on what kind of clothing it should expect each individual to be wearing. Once each individual's appearance is documented, the intelligent robot can then recognize and provide assistance to specific humans.
###  Random Line
The robot will need to recognize people in real time and the first step of this process is to generate a random line on a live image or image stream. The purpose of this is to define a boundary where the colors in an image to differ. This could define the boundary between a person's pants and shirt or the outfit is monochromatic, it will define the boundary between the person's outfit and skin.
![A-Probabilistic-Framework-for-Short-Term-Person-Recognition-random_line](https://raw.githubusercontent.com/julianweisbord/A-Probabilistic-Framework-for-Short-Term-Person-Recognition/master/data_capture/manipulated_images/line.jpeg)
###  Color Histograms
Next we can create color histograms, to acquire info about what colors are most present in the live image data.
![A-Probabilistic-Framework-for-Short-Term-Person-Recognition-color_histograms](https://raw.githubusercontent.com/julianweisbord/A-Probabilistic-Framework-for-Short-Term-Person-Recognition/master/data_capture/manipulated_images/color_histograms.png)
##  Body Classification

##  Dependencies
pip install -r dependencies.txt
