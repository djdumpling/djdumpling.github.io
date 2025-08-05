Hey! I'm Alex Wa, a 2nd year Math and CS major at Yale and a YES Scholar. My research interests span RL, NLP, and interpretability, but I've also done research in algebraic topology, abstract algebra, and in the BME space.

In my free time, I enjoy [drawing](https://www.instagram.com/alex_wa_art/) and exploring other ML and math disciplines.

<div class="profile-container">
<img src="public/pfp.jpg" alt="Alex Wa" class="profile-image">
<div class="profile-content">
</div>
</div>

## Posts

{% for post in site.posts %}
  <div class="post-entry">
    <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
    <p class="post-date">{{ post.date | date: "%B %d, %Y" }}</p>
    {% if post.summary %}
      <p class="post-summary">{{ post.summary }}</p>
    {% endif %}
  </div>
{% endfor %}