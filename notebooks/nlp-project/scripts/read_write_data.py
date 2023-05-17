import codecs

def read_processed_data(file_name):
    """
    read in conll file
    
    :param file_name: path to read from
    :yields: list of words, tags, labels, domain for each sentence
    """
    current_words = []
    current_tags = []
    current_labels = []

    for line in codecs.open(file_name, encoding='UTF-8'):
        line = line.strip()
        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')
            word, tag, label, domain = tok

            current_words.append(word)
            current_tags.append(tag)
            current_labels.append(label)
        else:
            if current_words:  # skip empty lines
                yield((current_words, current_tags, current_labels, domain))
            current_words = []
            current_tags = []
            current_labels = []

    # check for last one
    if current_tags != [] and not raw:
        yield((current_words, current_tags, current_labels, domain))

def load_data(file_name):
    words = []
    tags = []
    labels= []
    domains=[]
    for word, tag, label, domain in read_processed_data(file_name):
        words.append(word)
        tags.append(tag)
        labels.append(label)
        domains.append(domain)
    return words, tags, labels, domains

def read_raw_data(file_name):
    """
    read in conll file
    
    :param file_name: path to read from
    :yields: list of words and labels for each sentence
    """
    current_words = []
    current_tags = []

    for line in codecs.open(file_name, encoding='UTF-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')
            word = tok[0]
            tag = tok[1]

            current_words.append(word)
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                yield((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        yield((current_words, current_tags))

def write_conll(df, path):
    with open(path, encoding='utf-8', mode='w') as f:
        for i, sentence in df.iterrows():
            words, isNE, labels, domain  = sentence
            for word, label, NE in zip(words,labels,isNE):
                f.write(word+'\t'+str(NE)+'\t'+label+'\t'+domain+'\n')
            f.write('\n')

def write_baseline_pred(df, path):
    with open(path, encoding='utf-8', mode='a') as file:
        for i, sentence in df.iterrows():
            words, labels = sentence
            for word, label in zip(words,labels):
                file.write(word+'\t'+label+'\n')
            file.write('\n')