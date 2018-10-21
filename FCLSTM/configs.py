import tensorflow as tf

tf.flags.DEFINE_string("embedding_size", 100, "input embedding size")
tf.flags.DEFINE_string("n_output", 100, "output embedding size")
# tf.flags.DEFINE_string("n_steps", 200, "word topic size")
tf.flags.DEFINE_string("epoches", 10, "training epoches")
tf.flags.DEFINE_string("word_size", 150, "n_steps")
tf.flags.DEFINE_string("batch_size", 128, "mini batch size")
tf.flags.DEFINE_string("n_hidden", 128, "n_hidden")
tf.flags.DEFINE_string("margin", 0.1, "margin")
tf.flags.DEFINE_string("learning_rate", 0.05, "learning_rate")
tf.flags.DEFINE_string("tag_size", 10, "n_step_tag")
tf.flags.DEFINE_string("optimizer", 'SGD', "SGD or Adagrad")
tf.flags.DEFINE_string("k", 5, "k-fold validation")
tf.flags.DEFINE_string("pool_size", 5, "service pool size for each training mashup")
tf.flags.DEFINE_string("phi", "WebService/model/service_phi.txt", "word topic vectors")
tf.flags.DEFINE_string("expansionfromservice", "WebService/expansionfromservice.txt", "expended service descriptions")
tf.flags.DEFINE_string("expansionfromserviceandmashup", "WebService/expansionfromserviceandmashup.txt", "expended service descriptions")
tf.flags.DEFINE_string("MashupsSentenceToken", "WebService/MashupsSentenceToken.txt", "mashup descriprions")
tf.flags.DEFINE_string("descriptionforFCLSTM", "WebService/descriptions_for_FCLSTM", "descriprions for FCLSTM")
tf.flags.DEFINE_string("vocabulary", "WebService/outcomes/vocabulary.txt", "vocabulary of all descriptions")
tf.flags.DEFINE_string("compositionnet", "WebService/APIsHostMashups.txt", "vocabulary of all descriptions")
tf.flags.DEFINE_string("serviceTagToken", "WebService/APIsTagToken.txt", "tags of all service descriptions")
tf.flags.DEFINE_string("MashupTagToken", "WebService/MashupsTagToken.txt", "tags of of all Mashup descriptions")


FLAGS = tf.flags.FLAGS
