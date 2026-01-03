---
layout: page
title: Ongoing
---

Posts that are works-in-progress and updated regularly.

{% assign ongoing_posts = site.posts | where: "ongoing", true %}
{% for post in ongoing_posts %}
<div class="post-item">
  <div class="post-date">{{ post.date | date: "%B %d, %Y" }}</div>
  <h3 class="post-title">
    <a class="post-link" href="{{ post.url | relative_url }}">
      {{ post.title | escape }}
    </a>
  </h3>
  {%- if post.tokens -%}
    <div class="post-meta">{{ post.tokens }} tokens</div>
  {%- endif -%}
  {%- if post.excerpt -%}
    <div class="post-summary">{{ post.excerpt }}</div>
  {%- endif -%}
</div>
{% endfor %}

