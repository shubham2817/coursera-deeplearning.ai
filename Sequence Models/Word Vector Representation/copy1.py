import numpy as np
from w2v_utils2 import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    
    distance = 0.0

    dot = np.dot(u,v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cosine_similarity = dot/(norm_u * norm_v)
    
    return cosine_similarity

father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
print(">>"*30)




def complete_analogy(word_a, word_b, word_c, word_to_vec_map):

    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    for w in words:        
        if w in [word_a, word_b, word_c] :
            continue

        cosine_sim = cosine_similarity(np.subtract(e_b,e_a), np.subtract(word_to_vec_map[w],e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word


triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print triad
    word_a, word_b, word_c = triad
    print complete_analogy(word_a, word_b, word_c,word_to_vec_map)
    # print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
print(">>"*30)







g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)
print(">>"*30)



print ('List of names and their similarities with constructed vector:')
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
print(">>"*30)





print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
print(">>"*30)








#### ERROR ----------------------------------------

'''
def neutralize(word, g, word_to_vec_map):

    e = None
    e_biascomponent = None
    e_debiased = None    
    return e_debiased


e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))




def equalize(pair, bias_axis, word_to_vec_map):

    w1, w2 = None
    e_w1, e_w2 = None

    mu = None

    mu_B = None
    mu_orth = None

    e_w1B = None
    e_w2B = None
        
    corrected_e_w1B = None
    corrected_e_w2B = None

    e1 = None
    e2 = None

    return e1, e2


print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
'''