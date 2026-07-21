import xml.etree.ElementTree as etree


class Converter:

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    def __init__(self, tree: etree.ElementTree):
        root = tree.getroot()
        if root is None:
            raise Exception("The element tree is None")
        self.header = root[0]
        self.content = root[1]

    def extract_header(self) -> tuple[str, str]:
        """Extracts the header of the XML and give back the title and abstract"""

        title_ele = self.header.find(".//tei:title[1]", namespaces=self.ns)
        abstract_ele = self.header.find(
            ".//tei:abstract[1]/tei:div/tei:p", namespaces=self.ns
        )
        title = "Not able to fetch the title"
        abstract = "Not able to fetch the abstract"
        if title_ele is not None:
            title = title_ele.text.strip() if title_ele.text else title
        if abstract_ele is not None:
            abstract = abstract_ele.text.strip() if abstract_ele.text else abstract

        return title, abstract

    def extract_content(self) -> str:
        """
        Extract the contents of the XML and gives back str which is in .md format
        """
        # path : text/body/[div] > (head,p)
        extracted_text = []
        divs = self.content.findall(".//tei:body/tei:div", namespaces=self.ns)
        for div in divs:
            heading = div[0].text if div[0].text else "Heading is not found"
            n = div[0].get("n", None)
            header_weight = n.count(".") + 1 if n is not None else 0
            header = f"\n{header_weight * '#'} {heading}\n"
            extracted_text.append(header)
            p_tags = div.findall(".//tei:p", self.ns)
            for p_tag in p_tags:
                if p_tag is not None:
                    text = (
                        "".join(p_tag.itertext()) + "\n"
                    )  # We are using iter over here cause in a p tag it might have some other
                    # nested tag like ref and in that case p.text alone won't give us all
                    # the text inside the p tag
                    extracted_text.append(text)

        return "".join(extracted_text)
