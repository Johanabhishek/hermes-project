// Blog posts configuration
export const blogPosts = [
    {
      id: "hidden-dangers-data-augmentation",
      title: "The Hidden Dangers of Data Augmentation: A Benchmark Story",
      date: "2025-08-08",
      excerpt: "A controlled experiment reveals that some popular data augmentation techniques can actually harm your model's performance.",
      content: `
  ## So, Data Augmentation is Always a Good Idea, Right? I Decided to Check.
  
  Anyone who's spent time in machine learning has heard the advice: if you want a better model, get more data...
  // You will need to copy the full blog post content we drafted here.
  `
    }
  ];
  
  // Helper functions
  export const getPostById = (id) => blogPosts.find(post => post.id === id);
  export const getAllPosts = () => blogPosts;
  