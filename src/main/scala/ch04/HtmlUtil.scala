package ch04

/**
 * Created by leorick on 2015/12/25.
 */
object HtmlUtil {
  def removeConcateTag(content:String, replacement: String) = {
    content.replaceAll("-", replacement).
      replaceAll("=", replacement)
  }

  def removeHtmlTag(content:String, replacement: String) = {
    content.replaceAll("""<(!--|\p{Alpha}|/\p{Alpha})+[^>]*>""", replacement)
  }

  def removePunct(content:String, replacement: String) = {
    content.replaceAll("""\p{Punct}""", replacement)
  }

  def removeLF(content:String, replacement: String) = {
    content.replaceAll("\n", replacement)
  }

  def getLoCaseHtmlBody(content:String) = {
    """<body>.*</body>""".r.findFirstIn(content.toLowerCase).getOrElse(content)
  }
}
