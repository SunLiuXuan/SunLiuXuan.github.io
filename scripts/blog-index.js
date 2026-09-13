'use strict';
const pagination = require('hexo-pagination');
hexo.extend.generator.register('index', function(locals) {
  const posts = locals.posts.sort('-date');
  posts.data = posts.toArray().slice().sort((a, b) =>
    (Number(b.top) || 0) - (Number(a.top) || 0) || b.date.valueOf() - a.date.valueOf());
  return pagination(this.config.index_generator.path || 'blogs', posts, {
    perPage: posts.length ? this.config.index_generator.per_page : 0,
    layout: ['index', 'archive'],
    format: (this.config.pagination_dir || 'page') + '/%d/',
    data: { __index: true }
  });
});

