# Project 1 feedback

### Rubric

 - Data exploration: 9/10
 - Model choice/exploration: 10/10
 - Code: 6/10
 - Report:
 
     **Content** 
     
     - Organization: 8/8
     - Objective description: 4/5
     - Analysis: 10/10
     - Results: 9/10
     
     **Writing**
     
     - Technical Correctness: 5/5
     - Tone: 2/2
     - Overall writing: 5/5
 
 - Total: 68/75
 
 
#### Dr. Kuchera's Notes:

**Data exploration:** Overall good. You made a bold statement that we don't care about eliminating features because their coefficients would be small (which may bot be true if we have overfitting), but then you added regularization later, which effectively does this. I think the logic is there, but maybe not motivated strongly in the report.

**Code:** While your code (I only found code in your Jupyter notebook) does have the dosctrings suggested by PEP-8, there are no inline comments. More importantly, much of the code in Jupyter could have been hidden from the reader via a user-defined module without loss of understanding (plotting code, imports, etc.). Also, you had two notebooks. I assumed the one that did not have an error was the correct one. I shouldn't have to guess though!

**Objective description** In terms of the goal, we are predicting wine quality using a regression model. Gradient descent is a way to fit you regression model. So, your project objective was not gradient descent.

**Results** Your final model should be easy to find. A clear statement of your final model (linear w/ no regularization) would help. Also, for a reader interested in making wine predictions, MSE may not be the final exciting product. Maybe the function with coefficients or some other useful summary of findings.

**Writing** The writing and flow of the post was great. I can imagine this as a very readable, successful blog post with some of the code moved to a module and a slightly stronger beginning and end.
 
 
#### Student Feedback:


