//! Reads an EPUB's spine into chapters of paragraphs.

use crate::text::Paragraph;
use epub::doc::{EpubDoc, NavPoint};
use scraper::{ElementRef, Html, Node, Selector};
use std::error::Error;
use std::path::Path;

const BLOCKS: [&str; 22] = [
    "address", "article", "blockquote", "body", "dd", "div", "dl", "dt", "figure", "figcaption",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "p", "pre", "section",
];
const SKIPPED: [&str; 8] = ["head", "img", "nav", "script", "style", "svg", "table", "aside"];
const SKIPPED_ROLES: [&str; 7] = [
    "noteref", "footnote", "endnote", "rearnote", "pagebreak", "toc", "doc-noteref",
];
/// Project Gutenberg's header and licence (`pg-boilerplate`) and contents lists
/// marked as `toc` aren't part of the story. The licence alone is about 18,000
/// characters, and sits in the same file as the last chapter.
const SKIPPED_CLASSES: [&str; 2] = ["pg-boilerplate", "toc"];

pub struct Chapter {
    pub title: String,
    pub paragraphs: Vec<Paragraph>,
}

pub struct Book {
    pub title: String,
    pub author: Option<String>,
    /// A JPEG or PNG, the formats both M4B players and Cast displays show.
    pub cover: Option<Vec<u8>>,
    pub chapters: Vec<Chapter>,
}

/// The MIME type and file extension of a JPEG or PNG, judged by its first bytes.
pub fn image_type(bytes: &[u8]) -> Option<(&'static str, &'static str)> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(("image/jpeg", "jpg"))
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(("image/png", "png"))
    } else {
        None
    }
}

/// Inlines entities declared in the DOCTYPE and removes it. HTML parsers don't
/// honour custom entities, and end the DOCTYPE at its first `>`, which would
/// spill the declarations into the text.
fn expand_doctype(source: &str) -> String {
    let Some(start) = source.find("<!DOCTYPE") else {
        return source.to_string();
    };
    let after = &source[start..];
    let Some(subset) = after.find('[').filter(|&open| {
        after.find('>').is_some_and(|close| open < close)
    }) else {
        return source.to_string();
    };
    let Some(end) = after.find("]>") else {
        return source.to_string();
    };

    let mut body = format!("{}{}", &source[..start], &after[end + 2..]);
    for declaration in after[subset..end].split("<!ENTITY").skip(1) {
        let mut parts = declaration.trim().splitn(2, char::is_whitespace);
        let (Some(name), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        let value = value.trim();
        let Some(value) = value
            .strip_prefix('"')
            .and_then(|value| value.split('"').next())
        else {
            continue;
        };
        body = body.replace(&format!("&{name};"), value);
    }
    body
}

fn is_skipped(element: &ElementRef) -> bool {
    let tag = element.value().name();
    if SKIPPED.contains(&tag) || element.value().classes().any(|class| SKIPPED_CLASSES.contains(&class)) {
        return true;
    }
    // Footnote markers are superscript links.
    if tag == "sup" && element.children().any(|child| {
        ElementRef::wrap(child).is_some_and(|child| child.value().name() == "a")
    }) {
        return true;
    }
    ["epub:type", "role"].iter().any(|attribute| {
        element.value().attr(attribute).is_some_and(|value| {
            value.split_whitespace().any(|role| SKIPPED_ROLES.contains(&role))
        })
    })
}

/// Some books style plain blocks as headings instead of using `h1` to `h6`.
fn is_heading(element: &ElementRef) -> bool {
    let tag = element.value().name();
    matches!(tag, "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
        || element.value().attr("role") == Some("heading")
        || element.value().classes().any(|class| {
            let class = class.to_lowercase();
            ["title", "heading", "chapnum"].iter().any(|name| class.contains(name))
        })
}

fn flush(current: &mut String, heading: bool, paragraphs: &mut Vec<Paragraph>) {
    let text = current.split_whitespace().collect::<Vec<_>>().join(" ");
    current.clear();
    // Drops rows of decorative stars, which have nothing to read.
    if text.chars().any(char::is_alphanumeric) {
        paragraphs.push(Paragraph { text, heading });
    }
}

fn walk(element: ElementRef, current: &mut String, heading: bool, paragraphs: &mut Vec<Paragraph>) {
    for child in element.children() {
        match child.value() {
            Node::Text(text) => current.push_str(text),
            Node::Element(_) => {
                let Some(child) = ElementRef::wrap(child) else {
                    continue;
                };
                if is_skipped(&child) {
                    continue;
                }
                let tag = child.value().name();
                if tag == "br" {
                    current.push(' ');
                } else if BLOCKS.contains(&tag) {
                    let child_heading = heading || is_heading(&child);
                    flush(current, heading, paragraphs);
                    walk(child, current, child_heading, paragraphs);
                    flush(current, child_heading, paragraphs);
                } else {
                    walk(child, current, heading, paragraphs);
                }
            }
            _ => {}
        }
    }
}

/// Extracts the readable paragraphs from one XHTML document.
fn extract_paragraphs(source: &str) -> Vec<Paragraph> {
    let document = Html::parse_document(&expand_doctype(source));
    let mut paragraphs = Vec::new();
    let Ok(selector) = Selector::parse("body") else {
        return paragraphs;
    };
    if let Some(body) = document.select(&selector).next() {
        let mut current = String::new();
        walk(body, &mut current, false, &mut paragraphs);
        flush(&mut current, false, &mut paragraphs);
    }
    paragraphs
}

fn flatten(points: &[NavPoint], out: &mut Vec<(String, String)>) {
    for point in points {
        let path = point.content.to_string_lossy();
        let path = path.split('#').next().unwrap_or_default().to_string();
        out.push((path, point.label.trim().to_string()));
        flatten(&point.children, out);
    }
}

/// Finds a Dublin Core value in a package file. Older books nest these in a
/// `<dc-metadata>` wrapper, which the epub crate doesn't look inside.
fn dublin_core(package: &str, name: &str) -> Option<String> {
    let opening = format!("<dc:{name}");
    let start = package.find(&opening)? + opening.len();
    let rest = &package[start..];
    let value = &rest[rest.find('>')? + 1..];
    let value = &value[..value.find('<')?];
    let document = Html::parse_fragment(value);
    let text = document.root_element().text().collect::<String>();
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    (!text.is_empty()).then_some(text)
}

/// Reads a book. Each spine document with readable text becomes one chapter.
pub fn read_book(path: &Path) -> Result<Book, Box<dyn Error>> {
    let mut doc = EpubDoc::new(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let root_file = doc.root_file.clone();
    let package = doc.get_resource_str_by_path(root_file).unwrap_or_default();
    let title = doc
        .get_title()
        .or_else(|| dublin_core(&package, "title"))
        .unwrap_or_else(|| "Untitled".to_string());
    let author = doc
        .mdata("creator")
        .map(|item| item.value.clone())
        .or_else(|| dublin_core(&package, "creator"));
    let cover = doc
        .get_cover()
        .map(|(bytes, _)| bytes)
        .filter(|bytes| image_type(bytes).is_some());
    let mut toc = Vec::new();
    flatten(&doc.toc, &mut toc);

    let mut chapters = Vec::new();
    for item in doc.spine.clone() {
        let is_nav = item.properties.as_deref().is_some_and(|value| value.contains("nav"));
        if !item.linear || is_nav {
            continue;
        }
        let Some((source, _)) = doc.get_resource_str(&item.idref) else {
            continue;
        };
        let paragraphs = extract_paragraphs(&source);
        if paragraphs.is_empty() {
            continue;
        }
        let resource = doc
            .resources
            .get(&item.idref)
            .map(|resource| resource.path.to_string_lossy().to_string())
            .unwrap_or_default();
        let toc_title = toc
            .iter()
            .find(|(path, _)| *path == resource)
            .map(|(_, label)| label.clone());
        let heading = paragraphs
            .iter()
            .find(|paragraph| paragraph.heading)
            .map(|paragraph| paragraph.text.clone());
        let title = toc_title
            .or(heading)
            .unwrap_or_else(|| format!("Section {}", chapters.len() + 1));
        chapters.push(Chapter { title, paragraphs });
    }
    if chapters.is_empty() {
        return Err(format!(
            "no readable text in {}; DRM-protected books can't be read",
            path.display()
        )
        .into());
    }
    Ok(Book {
        title,
        author,
        cover,
        chapters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styled_blocks_count_as_headings() {
        let paragraphs = extract_paragraphs(
            r#"<html><body><div class="title_box"><div class="chapnum">Chapter II</div>
            <div class="chaptitle">The Pool</div></div><div class="body">Curiouser!</div></body></html>"#,
        );
        let found: Vec<(&str, bool)> =
            paragraphs.iter().map(|p| (p.text.as_str(), p.heading)).collect();
        assert_eq!(found, vec![("Chapter II", true), ("The Pool", true), ("Curiouser!", false)]);
    }

    const SOURCE: &str = r##"<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html [
<!ENTITY O "&#x201C;">
<!ENTITY C "&#x201D;">
]>
<html><head><title>Not this</title></head><body>
<div class="title_box"><h2>Chapter I</h2><h2>The Start</h2><hr/></div>
<p><img src="a.gif" alt="Not this either"/> Hello &O;there&C; <i>you</i>.<sup><a href="#n1">1</a></sup></p>
<pre>  *   *   *  </pre>
<div class="poem"><span>Line one,</span><br/><span>line two.</span></div>
<aside epub:type="footnote"><p>A footnote.</p></aside>
</body></html>"##;

    #[test]
    fn keeps_only_what_should_be_read() {
        let paragraphs = extract_paragraphs(SOURCE);
        let texts: Vec<(&str, bool)> = paragraphs
            .iter()
            .map(|paragraph| (paragraph.text.as_str(), paragraph.heading))
            .collect();
        assert_eq!(
            texts,
            vec![
                ("Chapter I", true),
                ("The Start", true),
                ("Hello \u{201C}there\u{201D} you.", false),
                ("Line one, line two.", false),
            ]
        );
    }

    #[test]
    fn skips_project_gutenberg_boilerplate_and_contents_lists() {
        // As in Project Gutenberg's EPUBs, whose header and licence share a
        // file with the book's opening and ending.
        let source = r##"<html><body>
<div class="pg-boilerplate pgheader header" id="pg-header"><p>The Project Gutenberg eBook of Moby Dick</p></div>
<p class="toc"><b>CONTENTS</b></p>
<blockquote><div><p class="toc"><a href="#c1" class="pginternal">Chapter 1. Loomings.</a></p></div></blockquote>
<h2>Chapter 1. Loomings.</h2>
<p>Call me Ishmael.</p>
<div class="pg-boilerplate pgheader footer" id="pg-footer"><p>Project Gutenberg is a registered trademark.</p></div>
</body></html>"##;
        let texts: Vec<String> = extract_paragraphs(source).into_iter().map(|p| p.text).collect();
        assert_eq!(texts, vec!["Chapter 1. Loomings.", "Call me Ishmael."]);
    }

    #[test]
    fn finds_metadata_in_older_package_files() {
        let package = r#"<metadata><dc-metadata><dc:title>Tom &amp; Jerry</dc:title>
            <dc:creator opf:role="aut">A  Writer</dc:creator></dc-metadata></metadata>"#;
        assert_eq!(dublin_core(package, "title").as_deref(), Some("Tom & Jerry"));
        assert_eq!(dublin_core(package, "creator").as_deref(), Some("A Writer"));
        assert_eq!(dublin_core(package, "publisher"), None);
    }

    #[test]
    fn documents_without_a_subset_are_untouched() {
        let source = "<!DOCTYPE html><html><body><p>Hi.</p></body></html>";
        assert_eq!(expand_doctype(source), source);
        assert_eq!(extract_paragraphs(source)[0].text, "Hi.");
    }
}
