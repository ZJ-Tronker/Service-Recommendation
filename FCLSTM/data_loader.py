import numpy as np
import os
import logging
import configs as config
import matplotlib.pyplot as plt 
import random
import gc

logging.getLogger().setLevel(logging.INFO)

'''
Generate the training samples with format: <mashup, positive_service, negative_service>
Time: 8-March-2018
'''

# load the word embedding of words
def load_embedding(filename, vocabulary, embedding_size):
    embeddings = []
    word2id = {}
    idx = 0
    with open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                idx += 1
                probs = line.strip().split(" ")
                if len(probs) != embedding_size:
                    logging.error("embedding error, index is:%s"%(idx))
                    continue
                embedding = [float(prob) for prob in probs[0:]]
                embeddings.append(embedding)
        except Exception as e:
            logging.error("load embedding Exception,", e)
        finally:
            rf.close()
    with open(vocabulary, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                ps = line.strip().split("=")
                word2id[ps[1]]=ps[0]
        except Exception as e:
            logging.error("load embedding Exception,", e)
        finally:
            rf.close()        
    
    logging.info("load embedding finish!")
    return embeddings, word2id

# word_embeddings = load_embedding(config.FLAGS.phi, config.FLAGS.embedding_size)

# load the descriptions and constrain the number of words to word_size
def load_descriptions(service_desc_file, mashup_desc_file, word_size):
    service_descs = []
    mashup_descs = []
    # read the service description
    with open(service_desc_file, mode="r", encoding="utf-8") as rf:
        bw = open(config.FLAGS.descriptionforFCLSTM + "/serviceDesc.txt", 'w')
        try:
            for line in rf.readlines():
                word_set = []
                sens = line.strip().split("    ")
                for sen in sens:
                    word_set += sen.strip().split(" ")
                # constrain the number of words in each description equal to word_size
                if(len(word_set) < word_size):
                    expand_times = int(word_size / len(word_set))
                    remain_wordnum = word_size % len(word_set)
                    word_set = word_set * expand_times
                    indexes = random.sample(range(0, len(word_set)), remain_wordnum)
                    word_set += [word_set[i] for i in indexes]
                elif(len(word_set) > word_size):
                    indexes = random.sample(range(0, len(word_set)), len(word_set) - word_size)
                    indexes.sort(reverse=True)
                    for idx in indexes:
                        word_set.remove(word_set[idx])
                service_descs.append(word_set) 
                
                bw.write(str(word_set).encode("gbk", 'ignore').decode("gbk", "ignore"))
                bw.write("\n")
        finally:
            rf.close()
        bw.close()
    # read the mashup descriptions
    with open(mashup_desc_file, mode="r", encoding="utf-8") as rf:
        bw = open(config.FLAGS.descriptionforFCLSTM + "/mashupDesc.txt", 'w')
        try:
            for line in rf.readlines():
                word_set = []
                sens = line.strip().split("    ")
                for sen in sens:
                    word_set += sen.strip().split(" ")
                # constrain the number of words in each description equal to word_size
                if(len(word_set) < word_size):
                    expand_times = int(word_size / len(word_set))
                    remain_wordnum = word_size % len(word_set)
                    word_set = word_set * expand_times
                    indexes = random.sample(range(0, len(word_set)), remain_wordnum)
                    word_set += [word_set[i] for i in indexes]
                elif(len(word_set) > word_size):
                    indexes = random.sample(range(0, len(word_set)), len(word_set) - word_size)
                    indexes.sort(reverse=True)
                    for idx in indexes:
                        word_set.remove(word_set[idx])
                mashup_descs.append(word_set)    
                
                bw.write(str(word_set).encode("gbk", 'ignore').decode("gbk", "ignore"))
                bw.write("\n")
        finally:
            rf.close()
        bw.close()   
    logging.info("load the service and mashup descriptions finished! service=%d, mashup=%d" % (len(service_descs), len(mashup_descs)))    
    return service_descs, mashup_descs 
        

# load_descriptions(config.FLAGS.expansionfromserviceandmashup, config.FLAGS.MashupsSentenceToken, 250)

# load each description as embeddings with shape = [number of descriptions, word_size * embedding_size]
def load_description_embeddings():
    service_embeddings = []
    mashup_embeddings = []
    service_descs, mashup_descs = load_descriptions(config.FLAGS.expansionfromserviceandmashup, config.FLAGS.MashupsSentenceToken, config.FLAGS.word_size)
    word_embeddings, word2id = load_embedding(config.FLAGS.phi, config.FLAGS.vocabulary, config.FLAGS.embedding_size)
    # generate embeddings of services
    for desc_s in service_descs:
        desc_embedding = [word_embeddings[int(word2id[word])] for word in desc_s]
#         print(len(desc_embedding))
        service_embeddings.append(desc_embedding)
    
    service_embeddings = np.array(service_embeddings)
    service_embeddings = np.reshape(service_embeddings, (len(service_descs), config.FLAGS.word_size * config.FLAGS.embedding_size))
    #generate embeddings of mashups
    for desc_m in mashup_descs:
        desc_embedding = [word_embeddings[int(word2id[word])] for word in desc_m]
        mashup_embeddings.append(desc_embedding)
    mashup_embeddings = np.array(mashup_embeddings)
    mashup_embeddings = np.reshape(mashup_embeddings, (len(mashup_descs), config.FLAGS.word_size * config.FLAGS.embedding_size))
    
    return service_embeddings, mashup_embeddings

def load_description_tag_embeddings():
    service_embeddings = []
    mashup_embeddings = []
    servicetag_embeddings = []
    mashuptag_embeddings = []
    
    service_descs, mashup_descs = load_descriptions(config.FLAGS.expansionfromserviceandmashup, config.FLAGS.MashupsSentenceToken, config.FLAGS.word_size)
    service_tags, mashup_tags = load_tags(config.FLAGS.serviceTagToken, config.FLAGS.MashupTagToken, config.FLAGS.tag_size)
    word_embeddings, word2id = load_embedding(config.FLAGS.phi, config.FLAGS.vocabulary, config.FLAGS.embedding_size)
    
    
    # generate embeddings of services
    for desc_s in service_descs:
        desc_embedding = [word_embeddings[int(word2id[word])] for word in desc_s]
#         print(len(desc_embedding))
        service_embeddings.append(desc_embedding)
    
    service_embeddings = np.array(service_embeddings)
    service_embeddings = np.reshape(service_embeddings, (len(service_descs), config.FLAGS.word_size * config.FLAGS.embedding_size))
    #generate embeddings of mashups
    for desc_m in mashup_descs:
        desc_embedding = [word_embeddings[int(word2id[word])] for word in desc_m]
        mashup_embeddings.append(desc_embedding)
    mashup_embeddings = np.array(mashup_embeddings)
    mashup_embeddings = np.reshape(mashup_embeddings, (len(mashup_descs), config.FLAGS.word_size * config.FLAGS.embedding_size))

    # generate embeddings of service tags
    for tags_s in service_tags:
        tag_embedding = [word_embeddings[int(word2id[word])] for word in tags_s]
        servicetag_embeddings.append(tag_embedding)
    servicetag_embeddings = np.array(servicetag_embeddings)
    servicetag_embeddings = np.reshape(servicetag_embeddings, (len(service_tags), config.FLAGS.tag_size * config.FLAGS.embedding_size))
    
    # generate embeddings of Mashup tags
    for tags_s in mashup_tags:
        tag_embedding = [word_embeddings[int(word2id[word])] for word in tags_s]
        mashuptag_embeddings.append(tag_embedding)
    mashuptag_embeddings = np.array(mashuptag_embeddings)
    mashuptag_embeddings = np.reshape(mashuptag_embeddings, (len(mashup_tags), config.FLAGS.tag_size * config.FLAGS.embedding_size))
    
    return service_embeddings, mashup_embeddings, servicetag_embeddings, mashuptag_embeddings

# service_embeddings, mashup_embeddings = load_description_embeddings()

def load_tags(serviceTagFile, MashupTagFile, tag_size):
    service_tags = []
    mashup_tags = []
    # read the service description
    with open(serviceTagFile, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                word_set = []
                sens = line.strip().split("    ")
                for sen in sens:
                    word_set += sen.strip().split(" ")
                # constrain the number of words in each description equal to word_size
                if(len(word_set) < tag_size):
                    expand_times = int(tag_size / len(word_set))
                    remain_wordnum = tag_size % len(word_set)
                    word_set = word_set * expand_times
                    indexes = random.sample(range(0, len(word_set)), remain_wordnum)
                    word_set += [word_set[i] for i in indexes]
                elif(len(word_set) > tag_size):
                    indexes = random.sample(range(0, len(word_set)), len(word_set) - tag_size)
                    indexes.sort(reverse=True)
                    for idx in indexes:
                        word_set.remove(word_set[idx])
                service_tags.append(word_set) 
        finally:
            rf.close()
    # read the mashup descriptions
    with open(MashupTagFile, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                word_set = []
                sens = line.strip().split("    ")
                for sen in sens:
                    word_set += sen.strip().split(" ")
                # constrain the number of words in each description equal to word_size
                if(len(word_set) < tag_size):
                    expand_times = int(tag_size / len(word_set))
                    remain_wordnum = tag_size % len(word_set)
                    word_set = word_set * expand_times
                    indexes = random.sample(range(0, len(word_set)), remain_wordnum)
                    word_set += [word_set[i] for i in indexes]
                elif(len(word_set) > tag_size):
                    indexes = random.sample(range(0, len(word_set)), len(word_set) - tag_size)
                    indexes.sort(reverse=True)
                    for idx in indexes:
                        word_set.remove(word_set[idx])
                mashup_tags.append(word_set)    
                
        finally:
            rf.close()
    

def load_compositionnet(filename):
    nets = {}
    with open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                ps = line.strip().split("    ")
                sid = ps[0]
                mids = ps[1].split(" ")
                for mid in mids:
                    if mid in nets:
                        nets.get(mid).append(sid)
                    else:
                        nets[mid]=[sid]
        finally:
            rf.close()
    return nets

# plot the service distribution of mashups
def comservice_distributions():
    # load the composition nets
    nets = load_compositionnet(config.FLAGS.compositionnet)
    mcount = {}
    for ss in nets.values():
        count = len(ss)
        if count in mcount:
            mcount[count] += 1
        else:
            mcount[count] = 1
    x, y = list(mcount.keys())[0:10], list(mcount.values())[0:10]
    plt.bar(x, y, 0.7, facecolor='deepskyblue')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title("Word Size Distribution of the Expanded Service Descriptions")
    plt.xlabel("# of APIs",fontsize=15)
    plt.ylabel("Percentage of Mashups",fontsize=15)
    plt.show()

# comservice_distributions()


def load_data():
    mashups = []
    services_pos = []
    services_neg = []
    # load the description embeddings
    service_embeddings, mashup_embeddings = load_description_embeddings()
    print(service_embeddings.shape)
    print(mashup_embeddings.shape)
    # load the composition nets
    nets = load_compositionnet(config.FLAGS.compositionnet)
    print("nets=%d"%(len(nets)))
    # split the training set and testing set
    test_size = int(mashup_embeddings.shape[0] / config.FLAGS.k) 
    print("test_size=%d"%test_size)
    training_samples = random.sample(range(0, mashup_embeddings.shape[0]), (mashup_embeddings.shape[0] - test_size))
    test_sample = [item for item in range(mashup_embeddings.shape[0]) if item not in training_samples]
    training_count = 0
    for mid in training_samples:
        print("%d/%d"%(training_count,len(training_samples)))
        if str(mid) not in nets: 
            continue
        training_count += 1
        member_ids = nets[str(mid)]
        
        while len(member_ids) > config.FLAGS.pool_size:
            member_ids.remove(member_ids[random.sample(range(len(member_ids)), 1)[0]])
#         print("member_id=%d"%len(member_ids))
        
        while len(member_ids) < config.FLAGS.pool_size:
#             print(member_ids[random.sample(range(len(member_ids)), 1)[0]])
            member_ids += [member_ids[random.sample(range(len(member_ids)), 1)[0]]]
#             print("member_id=%d"%len(member_ids))
#         print("member_id=%d"%len(member_ids))
        neg_candidates = [item for item in range(service_embeddings.shape[0]) if item not in member_ids] 
        neg_samples = random.sample(neg_candidates, config.FLAGS.pool_size)
        # obtain the pos and neg services
#         print(member_ids)
        members = np.array([np.reshape(service_embeddings[int(i)], (config.FLAGS.word_size, config.FLAGS.embedding_size)) for i in member_ids])
        non_members = np.array([np.reshape(service_embeddings[i], (config.FLAGS.word_size, config.FLAGS.embedding_size)) for i in neg_samples]) 
#         service_embeddings[neg_samples]
        pair_mashups = np.array([np.reshape(mashup_embeddings[mid], (config.FLAGS.word_size, config.FLAGS.embedding_size)) for i in range(config.FLAGS.pool_size)])
        # generate all the training instances for the current mashup
        mashups.append(pair_mashups)
        services_pos.append(members)
        services_neg.append(non_members)
    mashups = np.concatenate(np.array(mashups), 0)
    services_pos = np.concatenate(np.array(services_pos), 0)
    services_neg = np.concatenate(np.array(services_neg), 0)
    return zip(mashups, services_pos, services_neg)
        

def load_data_withtags():
    mashups = []
    services_pos = []
    services_neg = []
    
    mashup_tags = []
    service_pos_tags = []
    service_neg_tags = []
    
    # load the description embeddings
    service_embeddings, mashup_embeddings, servicetag_embeddings, mashuptag_embeddings = load_description_tag_embeddings()
    print(service_embeddings.shape)
    print(servicetag_embeddings.shape)
    print(mashup_embeddings.shape)
    print(mashuptag_embeddings.shape)
    # load the composition nets
    nets = load_compositionnet(config.FLAGS.compositionnet)
    print("nets=%d"%(len(nets)))
    # split the training set and testing set
    test_size = int(mashup_embeddings.shape[0] / config.FLAGS.k) 
    print("test_size=%d"%test_size)
    training_samples = random.sample(range(0, mashup_embeddings.shape[0]), (mashup_embeddings.shape[0] - test_size))
    test_sample = [item for item in range(mashup_embeddings.shape[0]) if item not in training_samples]
    training_count = 0
    for mid in training_samples:
        print("%d/%d"%(training_count,len(training_samples)))
        if str(mid) not in nets: 
            continue
        training_count += 1
        member_ids = nets[str(mid)]
        
        while len(member_ids) > config.FLAGS.pool_size:
            member_ids.remove(member_ids[random.sample(range(len(member_ids)), 1)[0]])
#         print("member_id=%d"%len(member_ids))
        
        while len(member_ids) < config.FLAGS.pool_size:
#             print(member_ids[random.sample(range(len(member_ids)), 1)[0]])
            member_ids += [member_ids[random.sample(range(len(member_ids)), 1)[0]]]
#             print("member_id=%d"%len(member_ids))
#         print("member_id=%d"%len(member_ids))
        neg_candidates = [item for item in range(service_embeddings.shape[0]) if item not in member_ids] 
        neg_samples = random.sample(neg_candidates, config.FLAGS.pool_size)
        # obtain the pos and neg services
#         print(member_ids)

        # descriptions
        members = np.array([np.reshape(service_embeddings[int(i)], (config.FLAGS.word_size, config.FLAGS.embedding_size)) for i in member_ids])
        non_members = np.array([np.reshape(service_embeddings[i], (config.FLAGS.word_size, config.FLAGS.embedding_size)) for i in neg_samples]) 
#         service_embeddings[neg_samples]
        pair_mashups = np.array([np.reshape(mashup_embeddings[mid], (config.FLAGS.word_size, config.FLAGS.embedding_size)) for i in range(config.FLAGS.pool_size)])
        # generate all the training instances for the current mashup
        mashups.append(pair_mashups)
        services_pos.append(members)
        services_neg.append(non_members)
        
        # tags
        pair_mashup_tags = np.array([np.reshape(mashuptag_embeddings[mid], (config.FLAGS.tag_size, config.FLAGS.embedding_size)) for i in range(config.FLAGS.pool_size)])
        member_tags = np.array([np.reshape(servicetag_embeddings[int(i)], (config.FLAGS.tag_size, config.FLAGS.embedding_size)) for i in member_ids])
        non_member_tags = np.array([np.reshape(servicetag_embeddings[i], (config.FLAGS.tag_size, config.FLAGS.embedding_size)) for i in neg_samples]) 

        mashup_tags.append(pair_mashup_tags)
        service_pos_tags.append(member_tags)
        service_neg_tags.append(non_member_tags)
        
    mashups = np.concatenate(np.array(mashups), 0)
    services_pos = np.concatenate(np.array(services_pos), 0)
    services_neg = np.concatenate(np.array(services_neg), 0)
    
    mashup_tags = np.concatenate(np.array(mashup_tags), 0)
    service_pos_tags = np.concatenate(np.array(service_pos_tags), 0)
    service_neg_tags = np.concatenate(np.array(service_neg_tags), 0)
    return zip(mashups, services_pos, services_neg), zip(mashup_tags, service_pos_tags, service_neg_tags)
    
# data = load_data()
# print(len(list(data)))
# mashups, services_pos, services_neg = load_data()
# print(mashups.shape)
# print(services_pos.shape)
# print(services_neg.shape)

    
def statisitcs(filename):
    desclen = {}
    totalnum = 0
    idx = 0
    with open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                sens = line.strip().split("    ")
                wordnum = 0
                for sen in sens:
                    wordnum += len(sen.strip().split(" "))
                desclen[idx] = wordnum
                totalnum += wordnum
                idx += 1
        finally:
            rf.close()
    desclen = sorted(desclen.items(), key=lambda d: d[1])
    return desclen, totalnum

# plot the word size distribution of expanded descriptions
def wordsizedistribution():
    # desclen = statisitcs(config.FLAGS.expansionfromservice)
    desclen, totalnum = statisitcs(config.FLAGS.expansionfromserviceandmashup)
    removedNum = 1
    # sam = [desclen[i] for i in range(len(desclen)-removedNum, len(desclen))]
    # print(sam)
    print("average number of words: %d" % (totalnum / len(desclen)))
    x = [i for i in range(len(desclen)-removedNum)]
    y = [elem[1] for elem in desclen[:-removedNum]]
    # print(x[0:10])
    # print(y[0:10])
    plt.bar(x, y, 0.7, facecolor='green')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([0,16000,0,280])  
    # plt.title("Word Size Distribution of the Expanded Service Descriptions")
    plt.xlabel("# of service descriptions",fontsize=15)
    plt.ylabel("Number of words",fontsize=15)
    plt.show()
    
# wordsizedistribution()
