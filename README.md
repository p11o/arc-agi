# ARC-AGI

## Algorithm

* check for patterns between test input and ouput (ex: size, location of boxes)
* check for patterns across inputs (ex: size of input)
* check for patterns across outputs (ex: size of input, location of boxes, color of outputs)
* Can use test inputs and outputs as lookup values (what color do i use for X). Mapping rule?
* Check for color similarities
* Check for shape similarities
* For solutions with clipping, i.e. a shape is clipped in the solution, do a 40x40 grid, and then clip to the appropriate size.

Propose a rule
Test rule against "tests"


-----
Prompt

Okay here me out.  I'm working on the ARG-AGI contest. I think I know what I want to do at a high level, but I want some feedback, so tell me if this makes sense.
The input of the arc-agi contest consists of questions, each with a couple training examples to figure out the rule.  The training examples have both inputs and outputs.  Each question also has a test, and that only includes the input.  the program then has to determine what the rule is based on the training entries, apply that to the test input, and generate the output (or solution) to the test input.
The key thing about this contest is then to figure out how to create some abstract representation of this rule so that it can be applied to the test question. The way I want to do this is to have a two headed model that takes the input and output of the questions, this then outputs a rule representation, similar to a VGG network. I also need another model that takes the input and the rule, and applies that rule to generate the "output".  This may make the most sense as a transformer.  I'm not sure how you can do this attention mechanism, but do what you think makes the most sense and implement this in pytorch.