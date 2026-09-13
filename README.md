# Personal homepage & blog

Hexo powers the homepage at `/`, Projects at `/projects/`, and the blog at `/blogs/`.
Existing dated article URLs, tags, categories, and archives are preserved.

## Develop

```sh
npm ci
npm run clean
npm run build
npm run server
```

Pushes to `main` use the existing GitHub Pages workflow.

## Edit content

- `source/_data/profile.yml`: homepage intro, social links, news, selected projects, all projects, and selected blogs.
- `source/_posts/`: blog posts and their post asset folders.
- `source/about/index.md`: About page content.
- `source/images/`: shared images such as avatar and project thumbnails.

## Edit layout and style

- `templates/home.ejs`: homepage layout.
- `templates/projects.ejs`: Projects page layout.
- `source/css/site.css`: shared site styles, light/dark mode, homepage, projects, blog lists, and article typography.
- `source/js/theme.js`: light/dark mode toggle.
- `themes/slx/layout/`: local Hexo theme templates for blog lists, posts, pages, tags, categories, and archives.

`scripts/blog-index.js` replaces the old index plugin, respects `index_generator.path`, retains numeric `top` sorting, and generates pagination at `/blogs/page/N/`.
`scripts/homepage.js` generates the custom homepage, Projects page, and `/blog/` redirect.
