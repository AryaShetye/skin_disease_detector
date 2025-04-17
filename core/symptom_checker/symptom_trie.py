class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.diseases = set()

class SymptomTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, symptom, disease):
        node = self.root
        for char in symptom.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.diseases.add(disease)

    def search(self, prefix):
        node = self.root
        for char in prefix.lower():
            if char in node.children:
                node = node.children[char]
            else:
                return set()
        return self._collect_diseases(node)

    def _collect_diseases(self, node):
        results = set(node.diseases)
        for child in node.children.values():
            results.update(self._collect_diseases(child))
        return results
