# Qualitative complementary-pair examples (49 tokens)

Selection rule: every final-test pair is assigned to the first matching category below using only the correct-image and swapped-image generations; within a category pairs are ordered by SHA-256(6304:pair_id) and the first 2 are shown. Category sizes are given so the examples can be read against how common each behaviour is.

* **grounded**: both questions right with their own images AND both follow the swapped image's answer
* **insensitive**: identical answers for the two images (the answer does not respond to the image)
* **half_right**: exactly one of the two questions right with its own image
* **responsive_wrong**: different answers for the two images but both wrong

## Annotation-derived grounding

**[grounded - 36 pairs] presence**: Does this image contain a zebra?  
images val2014/184359 / val2014/454414; references 'no' / 'yes'  
- Correct: 'no' / 'yes' (NLL 0.00 / 0.00)
- Compl. swap: 'yes' / 'no' (NLL 8.00 / 6.81)
- Mismatched: 'no' / 'no' (NLL 0.01 / 4.33)
- No image: 'no' / 'no' (NLL 0.19 / 1.78)

**[grounded - 36 pairs] presence**: Does this image contain a pizza?  
images val2014/16995 / val2014/187887; references 'no' / 'yes'  
- Correct: 'no' / 'yes' (NLL 0.00 / 0.01)
- Compl. swap: 'yes' / 'no' (NLL 5.19 / 6.63)
- Mismatched: 'no' / 'no' (NLL 0.16 / 1.92)
- No image: 'yes' / 'yes' (NLL 0.75 / 0.64)

**[insensitive - 8 pairs] presence**: Does this image contain an orange?  
images val2014/208566 / val2014/554980; references 'no' / 'yes'  
- Correct: 'yes' / 'yes' (NLL 0.88 / 0.00)
- Compl. swap: 'yes' / 'yes' (NLL 6.33 / 0.54)
- Mismatched: 'no' / 'no' (NLL 0.04 / 3.17)
- No image: 'no' / 'no' (NLL 0.52 / 0.91)

**[insensitive - 8 pairs] counting**: How many birds can you see?  
images val2014/475223 / val2014/60456; references '2' / '1'  
- Correct: '1' / '1' (NLL 2.47 / 0.11)
- Compl. swap: '1' / '1' (NLL 2.49 / 0.11)
- Mismatched: '1' / '1' (NLL 2.07 / 0.16)
- No image: '1' / '1' (NLL 1.59 / 0.95)

**[half_right - 11 pairs] counting**: How many horses can you see?  
images val2014/507719 / val2014/354491; references '4' / '2'  
- Correct: '3' / '2' (NLL 1.45 / 0.04)
- Compl. swap: '2' / '3' (NLL 9.14 / 2.14)
- Mismatched: '1' / '1' (NLL 6.68 / 2.73)
- No image: '1' / '1' (NLL 2.13 / 1.30)

**[half_right - 11 pairs] counting**: How many ties can you see?  
images val2014/189504 / val2014/440522; references '3' / '1'  
- Correct: '2' / '1' (NLL 1.60 / 0.40)
- Compl. swap: '1' / '2' (NLL 5.15 / 1.62)
- Mismatched: '2' / '4' (NLL 3.06 / 2.44)
- No image: '1' / '1' (NLL 2.77 / 0.55)

**[responsive_wrong - 9 pairs] counting**: How many sinks can you see?  
images val2014/376310 / val2014/151651; references '3' / '4'  
- Correct: '2' / '3' (NLL 3.88 / 1.18)
- Compl. swap: '3' / '2' (NLL 0.79 / 6.54)
- Mismatched: '1' / '1' (NLL 4.52 / 5.85)
- No image: '1' / '1' (NLL 2.67 / 4.33)

**[responsive_wrong - 9 pairs] counting**: How many scissors can you see?  
images val2014/336046 / val2014/425905; references '2' / '1'  
- Correct: '1' / '2' (NLL 0.81 / 1.13)
- Compl. swap: '2' / '1' (NLL 0.45 / 0.68)
- Mismatched: '1' / '1' (NLL 2.83 / 0.83)
- No image: '1' / '1' (NLL 1.51 / 0.45)


## Natural VQAv2

**[grounded - 33 pairs] other**: What room is this?  
images val2014/32990 / val2014/554523; references 'bathroom' / 'kitchen'  
- Correct: 'bathroom' / 'kitchen' (NLL 0.12 / 0.58)
- Compl. swap: 'kitchen' / 'bathroom' (NLL 0.77 / 2.90)
- Mismatched: 'airport' / 'airport' (NLL 3.05 / 3.00)
- No image: 'hall' / 'hall' (NLL 2.57 / 2.12)

**[grounded - 33 pairs] other**: What animals are in the field?  
images val2014/162711 / val2014/463588; references 'cows' / 'horses'  
- Correct: 'cows' / 'horses' (NLL 0.32 / 0.20)
- Compl. swap: 'horses' / 'cows' (NLL 2.20 / 2.33)
- Mismatched: 'chicken' / 'donkey' (NLL 1.88 / 2.80)
- No image: 'bear' / 'bear' (NLL 3.08 / 3.29)

**[insensitive - 44 pairs] yes/no**: Is there more than one elephant?  
images val2014/210708 / val2014/214989; references 'yes' / 'no'  
- Correct: 'yes' / 'yes' (NLL 0.17 / 1.41)
- Compl. swap: 'yes' / 'yes' (NLL 0.31 / 1.98)
- Mismatched: 'no' / 'no' (NLL 2.81 / 0.14)
- No image: 'no' / 'no' (NLL 1.10 / 0.42)

**[insensitive - 44 pairs] yes/no**: Is this a secluded area?  
images val2014/386661 / val2014/506942; references 'yes' / 'no'  
- Correct: 'yes' / 'yes' (NLL 0.25 / 1.50)
- Compl. swap: 'yes' / 'yes' (NLL 0.25 / 1.50)
- Mismatched: 'no' / 'no' (NLL 0.78 / 0.62)
- No image: 'no' / 'no' (NLL 1.12 / 0.47)

**[half_right - 10 pairs] other**: What is around the teddy bears neck?  
images val2014/219344 / val2014/563758; references 'tie' / 'bandana'  
- Correct: 'tie' / 'band' (NLL 0.26 / 4.42)
- Compl. swap: 'band' / 'tie' (NLL 4.01 / 4.66)
- Mismatched: 'tents' / 'tents' (NLL 7.47 / 7.28)
- No image: 'bear' / 'bear' (NLL 5.91 / 4.48)

**[half_right - 10 pairs] other**: What color is the drink?  
images val2014/100553 / val2014/565941; references 'green' / 'brown'  
- Correct: 'green' / 'blue' (NLL 0.80 / 2.23)
- Compl. swap: 'blue' / 'green' (NLL 2.78 / 6.14)
- Mismatched: 'blue' / 'blue' (NLL 1.68 / 4.19)
- No image: 'blue' / 'blue' (NLL 2.40 / 4.54)

**[responsive_wrong - 9 pairs] other**: What color is the camera?  
images val2014/404944 / val2014/104982; references 'orange' / 'black'  
- Correct: 'red' / 'white' (NLL 3.28 / 1.92)
- Compl. swap: 'white' / 'red' (NLL 4.25 / 1.63)
- Mismatched: 'black' / 'black' (NLL 5.06 / 1.45)
- No image: 'red' / 'red' (NLL 4.61 / 1.52)

**[responsive_wrong - 9 pairs] other**: What is the main color of this?  
images val2014/122994 / val2014/128441; references 'red' / 'yellow'  
- Correct: 'blue' / 'red' (NLL 1.47 / 3.81)
- Compl. swap: 'red' / 'blue' (NLL 0.70 / 2.35)
- Mismatched: 'white' / 'white' (NLL 2.01 / 4.65)
- No image: 'red' / 'red' (NLL 1.03 / 3.61)
