#-*- coding: utf-8 -*-




class Coreano:
    def __init__(self, answers):
        self.con = ''
        self.vow = ''
        self.encon = ''
        #self.mix = ''
        self.last = ''
        self.answers = answers
        self.consonants = ['r','R','s','e','E','f','a','q','Q','t','T','d','w','W',
           'c','z','x','v','g']
        self.vowels = ['k','o','i','O','j','p','u','P','h','hk','ho','hl','y','n',
            'nj','np','nl','b','m','ml','l']
        self.endConsonants = ['','r','R','rt','s','sw','sg','e','f','fr','fa','fq','ft','fx','fv',
            'fg','a','q','qt','t','T','d','w','c','z','x','v','g']

        self.konsonants = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        self.doubleconsonants = ['R','E','Q','W','T', 'P', 'O']
        #doubleconsonants must include shifted vowels ex:ㅖ


    #charin would be pygame key name ( or some other thing)
    def takeChar(self, charin):

        #if space has been entered we must flush no matter what.
        # But rather than using this code below, I'll just call flush() at the feedback itself.
        #if charin == 'space':
        #    self.flush()

        #shift should only work on consonants WITH double consonants
        if charin not in self.doubleconsonants:
            charin = charin.lower()

        #nothing yet.
        if self.con == '':
            if charin in self.consonants:
                self.con = charin
                self.last += charin
                self.addone(self.unicon(charin))  # print

            elif charin in self.vowels:
                self.addone(self.univow(charin))  # print
                self.flush()

        #consoonant filled in, but vowel is empty
        elif self.vow == '':
            #con + vowel. always valid.
            if charin in self.vowels:
                self.vow = charin
                self.backone()
                self.addone(self.unikor())  # print
            #con + con? must start anew.rkr
            elif charin in self.consonants:
                self.flush()
                self.con = charin
                self.addone(self.unicon(charin))  # print

        # con and vow filled in, no end consonant yet.
        elif self.encon == '':
            #another vowel!?
            if charin in self.vowels:
                mix = self.vow + charin
                #if two vowels form a single valid vowel...
                if mix in self.vowels:
                    self.vow = mix
                    self.backone()
                    self.addone(self.unikor())  # print
                #if not, new input inc. but no valid mix can be made from this. so just print and move on.
                else:
                    self.flush()
                    self.addone(self.univow(charin))
            #finally we have an end consonant! change current text by erasing one..
            elif charin in self.endConsonants:
                self.encon = charin
                self.backone()
                self.addone(self.unikor())  # print

            elif charin in self.consonants: # ex)저 -> 저ㅉ
                self.flush()
                self.con = charin
                self.addone(self.unicon(charin))  # print
        # lastly, we already have end consonants, and we've received 1) vowel 2) consonant
        else:
            # if end consonant is made of only one type of char, remove that end consonant, flush, put it in consonant,
            #  and make a different combination with the new vowel!
            if charin in self.vowels:
                encon_type = len(list(set(self.encon)))
                if encon_type < 2:
                    temp_con = self.encon
                    self.encon = ''
                    self.backone()
                    self.addone(self.unikor())  # print
                    self.flush()
                    self.con = temp_con
                    self.vow = charin
                    self.addone(self.unikor())  # print
                #if it's made of two chars, split them!
                else:
                    temp_encon = self.encon[:1]
                    temp_con = self.encon[-1]
                    self.encon = temp_encon
                    self.backone()
                    self.addone(self.unikor())  # print
                    self.flush()
                    self.con = temp_con
                    self.vow = charin
                    self.addone(self.unikor())  # print
            # end consonants already filled in, but another consonant coming in? another two scenarios!
            elif charin in self.endConsonants:
                mix = self.encon + charin
                #new consonant forms an endConsonant with the old one
                if mix in self.endConsonants:
                    self.encon = mix
                    self.backone()
                    self.addone(self.unikor())  # print
                    #self.flush()  #dont flush here.. ex)출발: 춟->+ㅏ->출바
                #not.
                else:
                    self.flush()
                    self.con = charin
                    self.addone(self.unicon(charin))  # print

            elif charin in self.consonants:
                self.flush()
                self.con = charin
                self.addone(self.unicon(charin))  # print










    def flush(self):
        self.con = ''
        self.vow = ''
        self.encon = ''
        self.last = ''


    def addone(self, charin):
        self.answers.append(charin)

    def backone(self):
        del self.answers[-1]

    def clearFlush(self):
        self.flush()
        self.last = ''


    def unicon(self, charin):
        consonant = self.consonants.index(charin)
        #return unichr(12593 + (consonant * 1 + 0) + 0)
        return self.konsonants[consonant]

    def univow(self, charin):
        vowel = self.vowels.index(charin)
        return unichr(12623 + (vowel + 0) * 1 + 0)

    def unikor(self):
        consonant = self.consonants.index(self.con)
        vowel = self.vowels.index(self.vow)
        endConsonant = self.endConsonants.index(self.encon)
        return unichr(44032 + (consonant * 21 + vowel) * 28 + endConsonant)




    #####
    # def korInput(self, charin):
    #     if charin in self.consonants:
    #         consonant = self.consonants.index(charin)
    #         #return unichr(44032 + (consonant * 21 + 0) * 28 + 0)
    #         #return unichr(int((44032 + (consonant + 21) * 28 + 0)/21/28))
    #     elif charin in self.vowels:
    #         vowel = self.vowels.index(charin)
    #         #return unichr(44032 + (0 * 21 + vowel) * 28 + 0)
    #         return unichr(12623 + (vowel + 0) * 1 + 0)

