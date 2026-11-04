# Qualitative complementary-pair examples (196 tokens (main))

Selection rule: every final-test pair is assigned to the first matching category below using only the correct-image and swapped-image generations; within a category pairs are ordered by SHA-256(6304:pair_id) and the first 2 are shown. Category sizes are given so the examples can be read against how common each behaviour is.

* **grounded**: both questions right with their own images AND both follow the swapped image's answer
* **insensitive**: identical answers for the two images (the answer does not respond to the image)
* **half_right**: exactly one of the two questions right with its own image
* **responsive_wrong**: different answers for the two images but both wrong

## Annotation-derived grounding

**[grounded - 40 pairs] counting**: How many horses can you see?  
images val2014/507719 / val2014/354491; references '4' / '2'  
- Correct: '4' / '2' (NLL 0.39 / 0.01)
- Compl. swap: '2' / '4' (NLL 8.67 / 2.12)
- Mismatched: '0' / '0' (NLL 7.22 / 4.84)
- No image: '1' / '1' (NLL 2.52 / 1.09)

**[grounded - 40 pairs] presence**: Does this image contain a zebra?  
images val2014/184359 / val2014/454414; references 'no' / 'yes'  
- Correct: 'no' / 'yes' (NLL 0.01 / 0.00)
- Compl. swap: 'yes' / 'no' (NLL 8.80 / 4.26)
- Mismatched: 'no' / 'no' (NLL 0.02 / 4.20)
- No image: 'no' / 'no' (NLL 0.04 / 3.68)

**[insensitive - 11 pairs] counting**: How many ties can you see?  
images val2014/189504 / val2014/440522; references '3' / '1'  
- Correct: '1' / '1' (NLL 1.13 / 0.08)
- Compl. swap: '1' / '1' (NLL 5.06 / 1.09)
- Mismatched: '2' / '1' (NLL 1.39 / 0.64)
- No image: '2' / '2' (NLL 2.13 / 1.08)

**[insensitive - 11 pairs] counting**: How many scissors can you see?  
images val2014/336046 / val2014/425905; references '2' / '1'  
- Correct: '1' / '1' (NLL 1.06 / 0.45)
- Compl. swap: '1' / '1' (NLL 1.07 / 0.54)
- Mismatched: '1' / '1' (NLL 1.65 / 0.65)
- No image: '1' / '1' (NLL 1.19 / 0.89)

**[half_right - 7 pairs] counting**: How many cakes can you see?  
images val2014/468784 / val2014/329542; references '1' / '4'  
- Correct: '3' / '4' (NLL 1.94 / 0.71)
- Compl. swap: '4' / '3' (NLL 3.38 / 4.03)
- Mismatched: '1' / '1' (NLL 0.79 / 4.46)
- No image: '1' / '1' (NLL 0.83 / 2.81)

**[half_right - 7 pairs] counting**: How many bowls can you see?  
images val2014/143859 / val2014/1180; references '3' / '1'  
- Correct: '4' / '1' (NLL 1.70 / 0.95)
- Compl. swap: '1' / '4' (NLL 1.69 / 3.65)
- Mismatched: '1' / '2' (NLL 4.46 / 1.68)
- No image: '1' / '1' (NLL 1.71 / 1.05)

**[responsive_wrong - 6 pairs] counting**: How many sinks can you see?  
images val2014/376310 / val2014/151651; references '3' / '4'  
- Correct: '2' / '3' (NLL 2.35 / 3.49)
- Compl. swap: '3' / '2' (NLL 0.03 / 5.92)
- Mismatched: '1' / '1' (NLL 4.50 / 5.86)
- No image: '1' / '1' (NLL 2.74 / 4.04)

**[responsive_wrong - 6 pairs] counting**: How many zebras can you see?  
images val2014/188173 / val2014/486046; references '3' / '2'  
- Correct: '4' / '1' (NLL 1.59 / 3.06)
- Compl. swap: '1' / '4' (NLL 6.17 / 4.67)
- Mismatched: '0' / '0' (NLL 2.93 / 1.63)
- No image: '0' / '0' (NLL 2.25 / 1.58)


## Natural VQAv2

**[grounded - 31 pairs] other**: What room is this?  
images val2014/32990 / val2014/554523; references 'bathroom' / 'kitchen'  
- Correct: 'bathroom' / 'kitchen' (NLL 0.07 / 0.42)
- Compl. swap: 'kitchen' / 'bathroom' (NLL 0.81 / 2.30)
- Mismatched: 'airport' / 'airport' (NLL 3.18 / 3.28)
- No image: 'room' / 'room' (NLL 3.01 / 2.61)

**[grounded - 31 pairs] other**: What animals are in the field?  
images val2014/162711 / val2014/463588; references 'cows' / 'horses'  
- Correct: 'cows' / 'horses' (NLL 0.25 / 0.22)
- Compl. swap: 'horses' / 'cows' (NLL 2.38 / 3.96)
- Mismatched: 'no' / 'donuts' (NLL 1.99 / 3.51)
- No image: 'bear' / 'bear' (NLL 2.46 / 3.18)

**[insensitive - 43 pairs] yes/no**: Is there more than one elephant?  
images val2014/210708 / val2014/214989; references 'yes' / 'no'  
- Correct: 'yes' / 'yes' (NLL 0.09 / 0.74)
- Compl. swap: 'yes' / 'yes' (NLL 0.66 / 2.59)
- Mismatched: 'no' / 'no' (NLL 2.81 / 0.02)
- No image: 'no' / 'no' (NLL 2.29 / 0.27)

**[insensitive - 43 pairs] yes/no**: Is this a secluded area?  
images val2014/386661 / val2014/506942; references 'yes' / 'no'  
- Correct: 'yes' / 'yes' (NLL 0.09 / 1.17)
- Compl. swap: 'yes' / 'yes' (NLL 0.37 / 2.41)
- Mismatched: 'yes' / 'yes' (NLL 0.44 / 1.04)
- No image: 'no' / 'no' (NLL 2.80 / 0.41)

**[half_right - 13 pairs] other**: What is around the teddy bears neck?  
images val2014/219344 / val2014/563758; references 'tie' / 'bandana'  
- Correct: 'tie' / 'neck' (NLL 0.39 / 5.63)
- Compl. swap: 'neck' / 'tie' (NLL 3.64 / 7.75)
- Mismatched: 'teddy' / 'teddy' (NLL 5.86 / 6.63)
- No image: 'bear' / 'bear' (NLL 6.73 / 4.89)

**[half_right - 13 pairs] other**: What is this person about to do?  
images val2014/266977 / val2014/488219; references 'surf' / 'walk'  
- Correct: 'surf' / 'run' (NLL 0.31 / 3.68)
- Compl. swap: 'run' / 'surf' (NLL 3.81 / 7.32)
- Mismatched: 'zoo' / 'zoo' (NLL 8.86 / 3.07)
- No image: 'do' / 'do' (NLL 13.65 / 6.38)

**[responsive_wrong - 9 pairs] number**: How many bird feeders do you see?  
images val2014/35887 / val2014/553790; references '1' / '0'  
- Correct: '3' / '1' (NLL 2.47 / 1.67)
- Compl. swap: '1' / '3' (NLL 0.81 / 6.92)
- Mismatched: '1' / '0' (NLL 1.00 / 0.64)
- No image: '0' / '0' (NLL 1.35 / 1.10)

**[responsive_wrong - 9 pairs] other**: What does the shirt say?  
images val2014/539555 / val2014/187196; references 'gray' / 'nothing'  
- Correct: 'white' / 'knight' (NLL 6.11 / 6.83)
- Compl. swap: 'knight' / 'white' (NLL 5.10 / 5.19)
- Mismatched: 'white' / 'no' (NLL 5.30 / 4.79)
- No image: 'shirt' / 'shirt' (NLL 6.06 / 6.19)
