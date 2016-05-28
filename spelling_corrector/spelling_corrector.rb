NWORDS = File.read('big.txt').scan(/[a-zA-Z]+/).each_with_object(Hash.new(0)) { |key, hash| hash[key.downcase] += 1 }
ALPHABET = ('a'..'z').to_a

def edits1(word)
  deletes = (0...word.size).map { |i| word[0...i] << word[i+1..-1] }
  transposes = (1...word.size).map { |i| word[0...i-1] << word[i-1..i].reverse << word[i+1..-1] }
  replaces = (0...word.size).to_a.product(ALPHABET).map { |i, char| word[0...i] << char << word[i+1..-1] }
  inserts = (0..word.size).to_a.product(ALPHABET).map { |i, char| word[0...i] << char << word[i..-1] }
  (deletes + transposes + replaces + inserts).uniq.instance_eval { empty? ? nil : self }
end

def known_edits2(word)
  edits1(word).flat_map { |ed| known(edits1(ed)) }.uniq.compact.instance_eval { empty? ? nil : self }
end

def known(words)
  words.keep_if { |w| NWORDS.has_key? w }.instance_eval { empty? ? nil : self }
end

def correct(word)
  (known([word]) || known(edits1(word)) || known_edits2(word) || [word]).max_by { |c| NWORDS[c] }
end

puts ['physix', 'convnience', 'recieved', 'majar', 'grandour'].map { |x| correct x }
