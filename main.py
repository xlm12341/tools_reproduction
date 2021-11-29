import text2text as t2t
t2t.Transformer.PRETRAINED_TRANSLATOR = "facebook/m2m100_418M" #Remove this line for the larger model
h = t2t.Handler(["Hello, World!"], src_lang="en") #Initialize with some text
h.tokenize() #[['▁Hello', ',', '▁World', '!']]
print(t2t.Handler(["Hello, World! [SEP] Hello, what?"]).answer()) #['World'])
print(t2t.Transformer.LANGUAGES+'\n')

# Sample texts
article_en = 'The Secretary-General of the United Nations says there is no military solution in Syria.'

notre_dame_str = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student - run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one - page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut."

bacteria_str = "Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats."

bio_str = "Biology is the science that studies life. What exactly is life? This may sound like a silly question with an obvious answer, but it is not easy to define life. For example, a branch of biology called virology studies viruses, which exhibit some of the characteristics of living entities but lack others. It turns out that although viruses can attack living organisms, cause diseases, and even reproduce, they do not meet the criteria that biologists use to define life."

tfidf_index = t2t.Handler([
                       article_en,
                       notre_dame_str,
                       bacteria_str,
                       bio_str
                       ]).tfidf(output="matrix")

search_results_tf1 = t2t.Handler().search(
    queries=["wonderful life", "university students"],
    index=tfidf_index)

search_results_tf2 = t2t.Handler().search(
    queries=["Earth creatures are cool", "United Nations"],
    index=tfidf_index)
t2t.Handler([
         "Let's go hiking tomorrow",
         "안녕하세요.",
         "돼지꿈을 꾸세요~~"
         ]).tokenize()