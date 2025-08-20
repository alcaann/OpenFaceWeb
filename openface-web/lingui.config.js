module.exports = {
  locales: ["en", "no"],
  catalogs: [
    {
      path: "src/locales/{locale}/messages",
      include: ["src/"]
    }
  ],
  format: "po"
}
