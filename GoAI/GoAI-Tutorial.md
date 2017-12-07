# What google can do, can we do too!
Develop a Go AI.
Go to https://github.com/codeuniversity/ai-workshop and download the folder GoAI.
Learn how to play go here:
https://online-go.com/learn-to-play-go
In the two emplacements RTF-files from Github, you can find the emplacement either the Black or White had to face.  In the Moves-file you can find the action either Black or White took to react on the emplacement at the same index.
GoEmplacemantsBlack[5] --> GoMovesBlack[5]
GoEmplacementWhite[124] --> GooMovesWhite[124]
The emplacements are saved as an Array with 361 elements, each either 0, 1 or -1. The value 0 stands for an empty space whether taken by you nor by the enemy. The value 1 stands for space you occupied and the value -1 for space your enemy occupied. The AI should, in the end, return the position it wants to place their stone.
So your mission should you choose to accept it is to create an AI that can beat Fabian in Go.
##What to do
###1. Define the Size of your network
###2. Write the Placeholder for in- and output
###3. Write the Variables for Weight and Biases
###4. Evaluate the input by the hidden layer
###5. Apply the activation function
###6. Evaluate the Acivation of the neurons of the hidden layer
###7. Return the output
###8. Make the AI learn with the given dataset