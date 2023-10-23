STAR_PROMPTS_CQA_HINT ='''Q: What do people use to absorb extra ink from a fountain pen?
Answer Choices:
(a) shirt pocket
(b) calligrapher’s hand
(c) inkwell
(d) desk drawer
(e) blotter (CORRECT)
A: The answer must be used to absorb extra ink. Blotters are designed to absorb liquids. Therefore, the answer is blotter (e).

Q: What home entertainment equipment requires cable?
Answer Choices:
(a) radio shack
(b) substation
(c) television (CORRECT)
(d) cabinet
(e) desk
A: The answer must require cable. Cable is used to provide satellite channels to televisions. Therefore, the answer is television (c).

Q: The fox walked from the city into the forest, what was it looking for?
Answer Choices:
(a) pretty flowers
(b) hen house
(c) natural habitat (CORRECT)
(d) storybook
(e) dense forest
A: The answer must be a reason for a fox to go into the forest. The forest is a fox’s natural habitat. Therefore, the answer is natural habitat (c).

Q: Sammy wanted to go to where the people were. Where might he go?
Answer Choices:
(a) populated areas (CORRECT)
(b) race track
(c) desert
(d) apartment
(e) roadblock
A: The answer must be a place with many people. Populated areas, by definition, have a lot of people. Therefore, the answer is populated areas (a).

Q: Where do you put your grapes just before checking out?
Answer Choices:
(a) mouth
(b) grocery cart (CORRECT)
(c) super market
(d) fruit basket
(e) fruit market
A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. Therefore, the answer is grocery cart (b).

Q: Google Maps and other highway and street GPS services have replaced what?
Answer Choices:
(a) united states
(b) mexico
(c) countryside
(d) atlas (CORRECT)
(e) oceans
A: The answer must be something that used to do what Google Maps and GPS services do, which is give directions. Atlases were also used to give directions. Therefore, the answer is atlas (d).

Q: Before getting a divorce, what did the wife feel who was doing all the work?
Answer Choices:
(a) harder
(b) anguish
(c) bitterness (CORRECT)
(d) tears
(e) sadness
A: The answer should be a feeling which would cause someone who was doing all the work to get divorced. If someone feels bitter towards their spouse, they are likely to want a divorce. Therefore, the answer is bitterness (c).
'''

STAR_PROMPTS_CQA ='''Q: What do people use to absorb extra ink from a fountain pen?
Answer Choices:
(a) shirt pocket
(b) calligrapher’s hand
(c) inkwell
(d) desk drawer
(e) blotter
A: The answer must be used to absorb extra ink. Blotters are designed to absorb liquids. Therefore, the answer is blotter (e).

Q: What home entertainment equipment requires cable?
Answer Choices:
(a) radio shack
(b) substation
(c) television
(d) cabinet
(e) desk
A: The answer must require cable. Cable is used to provide satellite channels to televisions. Therefore, the answer is television (c).

Q: The fox walked from the city into the forest, what was it looking for?
Answer Choices:
(a) pretty flowers
(b) hen house
(c) natural habitat
(d) storybook
(e) dense forest
A: The answer must be a reason for a fox to go into the forest. The forest is a fox’s natural habitat. Therefore, the answer is natural habitat (c).

Q: Sammy wanted to go to where the people were. Where might he go?
Answer Choices:
(a) populated areas
(b) race track
(c) desert
(d) apartment
(e) roadblock
A: The answer must be a place with many people. Populated areas, by definition, have a lot of people. Therefore, the answer is populated areas (a).

Q: Where do you put your grapes just before checking out?
Answer Choices:
(a) mouth
(b) grocery cart
(c) super market
(d) fruit basket
(e) fruit market
A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. Therefore, the answer is grocery cart (b).

Q: Google Maps and other highway and street GPS services have replaced what?
Answer Choices:
(a) united states
(b) mexico
(c) countryside
(d) atlas
(e) oceans
A: The answer must be something that used to do what Google Maps and GPS services do, which is give directions. Atlases were also used to give directions. Therefore, the answer is atlas (d).

Q: Before getting a divorce, what did the wife feel who was doing all the work?
Answer Choices:
(a) harder
(b) anguish
(c) bitterness
(d) tears
(e) sadness
A: The answer should be a feeling which would cause someone who was doing all the work to get divorced. If someone feels bitter towards their spouse, they are likely to want a divorce. Therefore, the answer is bitterness (c).
'''

PROMPTS_GSM8K = '''Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
Reasoning: A large pizza has 16 slices, so 2 large pizzas have 32 slices. A small pizza has 8 slices, so 2 small pizzas have 16 slices. If we add 32 slices and 16 slices, we get 48 slices. Therefore, if Albert eats it all, he will eat 48 slices of pizza in that day.
Answer: 48

Question: Mary does her grocery shopping on Saturday. She does her shopping only at a specific store where she is allowed a credit of $100, which must be paid in full before her next shopping trip. That week she spent the full credit limit and paid $15 of it on Tuesday and $23 of it on Thursday. How much credit will Mary need to pay before her next shopping trip?
Reasoning: Mary spends her entire credit limit of $100 on Saturday. On Tuesday, she pays $15 towards her debt. On Thursday, she pays $23 towards her debt. This leaves her with a remaining balance of $100 - $15 - $23, which is equal to $62.
Answer: 62

Question: Ralph is going to practice playing tennis with a tennis ball machine that shoots out tennis balls for Ralph to hit. He loads up the machine with 175 tennis balls to start with. Out of the first 100 balls, he manages to hit 2/5 of them. Of the next 75 tennis balls, he manages to hit 1/3 of them. Out of all the tennis balls, how many did Ralph not hit?
Reasoning: Ralph hits 2/5 of the first 100 balls, so he hits 40 balls. Then, Ralph hits 1/3 of the next 75 balls, so he hits 25 more balls. In total, Ralph hits 40 + 25 = 65 balls. Finally, we know that Ralph started with 175 balls, so 175 - 65 = 110 balls not hitted.
Answer: 110
'''

PROMPTS_SVAMP = """Question: Paul had 50 books. After buying some in a garage sale he had 151 left. How many books did he buy?
Reasoning: The number of books Paul bought can be found by subtracting the final number of books from the initial number of books: 151 - 50 = 101. Therefore, Paul bought 101 books in the garage sale.
Answer: 101

Question: Luke played a trivia game and scored 154 points. If he gained the 11 points in each round. How many rounds did he play?
Reasoning: We need to divide Luke's total score by the number of points he gained in each round. Therefore, the number of rounds Luke played is 154 / 11 = 14.
Answer: 14

Question: Julia played tag with 17 kids on monday, 15 kids on tuesday and 2 kids on wednesday. How many kids did she play with altogether?
Reasoning: To find the total number of kids Julia played with, we need to add the number of kids she played with on each day. Therefore, the total number of kids Julia played with is 17 + 15 + 2 = 34.
Answer: 34
"""

PROMPTS_MultiArith = """Question: There are 64 students trying out for the school's trivia teams. If 36 of them didn't get picked for the team and the rest were put into 4 groups, how many students would be in each group?
Reasoning: The number of students who got picked for the team is 64 - 36 = 28. To find how many students would be in each group, we need to divide the number of students by the number of groups, which is 28 / 4 = 7. Therefore, there will be 7 students in each group. 
Answer: 7

Question: Cody bought 7 boxes of chocolate candy and 3 boxes of caramel candy. If each box has 8 pieces inside it, how much candy did he have total?
Reasoning: First, we need to find the total number of boxes Cody bought, which is 7 + 3 = 10 boxes. Then, we can multiply the number of boxes by the number of pieces of candy in each box to find the total amount of candy. Therefore, Cody had 10 x 8 = 80 pieces of candy in total.
Answer: 80

Question: For Halloween Robin scored 23 pieces of candy. She ate 7 pieces the first night and then her sister gave her 21 more pieces. How many pieces of candy does Robin have now?
Reasoning: We need to add the number of pieces of candy she had after the first night to the number of pieces her sister gave her. Therefore, the total number of pieces of candy Robin has now is 23 - 7 + 21 = 37.
Answer: 37
"""

PROMPTS_SQA = """Question: Are chinchillas cold-blooded?
Reasoning: Chinchillas are rodents, which are mammals. All mammals are warm-blooded. So, the answer is No.
Answer: No

Question: Would Janet Jackson avoid a dish with ham?
Reasoning: Janet Jackson follows an Islamic practice. Islamic culture avoids eating pork. Ham is made from pork. So, the answer is Yes.
Answer: Yes

Question: Can a honey bee sting a human more than once?
Reasoning: Human skin is tough, and the bee’s stinger gets lodged in the skin. The stinger becomes separated from the bee which dies soon after. So, the answer is No.
Answer: No

Question: Is average number of peas in a pod enough commas for a billion?
Reasoning: The average number of peas in a pod is 6 or 7. A billion is a number that has only 3 commas in it. So, the answer is Yes.
Answer: Yes
"""

PROMPTS_CQA ='''Question: What do people use to absorb extra ink from a fountain pen?
Answer Choices:
(a) shirt pocket
(b) calligrapher’s hand
(c) inkwell
(d) desk drawer
(e) blotter
Answer: The answer must be used to absorb extra ink. Blotters are designed to absorb liquids. Therefore, the answer is blotter (e).

Question: What home entertainment equipment requires cable?
Answer Choices:
(a) radio shack
(b) substation
(c) television
(d) cabinet
(e) desk
Answer: The answer must require cable. Cable is used to provide satellite channels to televisions. Therefore, the answer is television (c).

Question: Sammy wanted to go to where the people were. Where might he go?
Answer Choices:
(a) populated areas
(b) race track
(c) desert
(d) apartment
(e) roadblock
Answer: The answer must be a place with many people. Populated areas, by definition, have a lot of people. Therefore, the answer is populated areas (a).

Question: Where do you put your grapes just before checking out?
Answer Choices:
(a) mouth
(b) grocery cart
(c) super market
(d) fruit basket
(e) fruit market
Answer: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. Therefore, the answer is grocery cart (b).

Question: Google Maps and other highway and street GPS services have replaced what?
Answer Choices:
(a) united states
(b) mexico
(c) countryside
(d) atlas
(e) oceans
Answer: The answer must be something that used to do what Google Maps and GPS services do, which is give directions. Atlases were also used to give directions. Therefore, the answer is atlas (d).
'''



BASELINE_PROMPTS = {
    "gsm8k": "Question: {question}\nAnswer:",
    'cqa': "Question: {question}\nAnswer Choices:{choices}\nAnswer:",
}
for task in ['svamp', 'multiarith', 'sqa']: BASELINE_PROMPTS[task] = BASELINE_PROMPTS['gsm8k']

BASELINE_PROMPT_FEWSHOT = {
    'gsm8k':
"""Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
Answer: 48

Question: Mary does her grocery shopping on Saturday. She does her shopping only at a specific store where she is allowed a credit of $100, which must be paid in full before her next shopping trip. That week she spent the full credit limit and paid $15 of it on Tuesday and $23 of it on Thursday. How much credit will Mary need to pay before her next shopping trip?
Answer: 62

Question: Ralph is going to practice playing tennis with a tennis ball machine that shoots out tennis balls for Ralph to hit. He loads up the machine with 175 tennis balls to start with. Out of the first 100 balls, he manages to hit 2/5 of them. Of the next 75 tennis balls, he manages to hit 1/3 of them. Out of all the tennis balls, how many did Ralph not hit?
Answer: 110
""",
    'cqa':
'''Question: What do people use to absorb extra ink from a fountain pen?
Answer Choices:
(a) shirt pocket
(b) calligrapher’s hand
(c) inkwell
(d) desk drawer
(e) blotter
Answer: (e)

Question: What home entertainment equipment requires cable?
Answer Choices:
(a) radio shack
(b) substation
(c) television
(d) cabinet
(e) desk
Answer: (c)

Question: Sammy wanted to go to where the people were. Where might he go?
Answer Choices:
(a) populated areas
(b) race track
(c) desert
(d) apartment
(e) roadblock
Answer: (a)

Question: Where do you put your grapes just before checking out?
Answer Choices:
(a) mouth
(b) grocery cart
(c) super market
(d) fruit basket
(e) fruit market
Answer: (b)

Question: Google Maps and other highway and street GPS services have replaced what?
Answer Choices:
(a) united states
(b) mexico
(c) countryside
(d) atlas
(e) oceans
Answer: (d)
''',
    'svamp':
'''Question: Paul had 50 books. After buying some in a garage sale he had 151 left. How many books did he buy?
Answer: 101

Question: Luke played a trivia game and scored 154 points. If he gained the 11 points in each round. How many rounds did he play?
Answer: 14

Question: Julia played tag with 17 kids on monday, 15 kids on tuesday and 2 kids on wednesday. How many kids did she play with altogether?
Answer: 34
''',
    'multiarith':
"""Question: There are 64 students trying out for the school's trivia teams. If 36 of them didn't get picked for the team and the rest were put into 4 groups, how many students would be in each group?
Answer: 7

Question: Cody bought 7 boxes of chocolate candy and 3 boxes of caramel candy. If each box has 8 pieces inside it, how much candy did he have total?
Answer: 80

Question: For Halloween Robin scored 23 pieces of candy. She ate 7 pieces the first night and then her sister gave her 21 more pieces. How many pieces of candy does Robin have now?
Answer: 37
""",
}