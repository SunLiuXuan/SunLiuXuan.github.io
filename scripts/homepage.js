'use strict';
const path = require('path');

function getWriting(profile, posts) {
  return (profile.writing || []).map(item => {
    const post = posts.find(entry => entry.slug === item.slug);
    return post ? {
      title: post.title,
      url: '/' + post.path,
      date: post.date.format('YYYY-MM-DD'),
      summary: item.summary
    } : null;
  }).filter(Boolean);
}

hexo.extend.generator.register('personal-home', async function(locals) {
  const profile = locals.data.profile;
  const posts = locals.posts.toArray();
  const writing = getWriting(profile, posts);
  const html = await this.render.render({ path: path.join(this.base_dir, 'templates/home.ejs') },
    { profile, writing, year: new Date().getFullYear(), siteUrl: this.config.url });
  const projects = await this.render.render({ path: path.join(this.base_dir, 'templates/projects.ejs') },
    { profile, year: new Date().getFullYear(), siteUrl: this.config.url });
  const blogRedirect = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=/blogs/"><link rel="canonical" href="/blogs/"><title>Blogs</title><p><a href="/blogs/">Blogs</a></p>';
  return [
    { path: 'index.html', data: html },
    { path: 'projects/index.html', data: projects },
    { path: 'blog/index.html', data: blogRedirect }
  ];
});
