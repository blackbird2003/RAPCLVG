(() => {
  let lightbox;
  let lightboxImage;

  function ensureLightbox() {
    if (lightbox) return;
    lightbox = document.createElement("div");
    lightbox.className = "image-lightbox";
    lightbox.setAttribute("role", "dialog");
    lightbox.setAttribute("aria-modal", "true");
    lightbox.setAttribute("aria-label", "Image preview");
    lightbox.innerHTML = '<button type="button" aria-label="Close image preview">&times;</button><img alt="">';
    lightboxImage = lightbox.querySelector("img");
    document.body.appendChild(lightbox);
    lightbox.addEventListener("click", (event) => {
      if (event.target === lightbox || event.target.tagName === "BUTTON") closeLightbox();
    });
  }

  function openLightbox(sourceImage) {
    ensureLightbox();
    lightboxImage.src = sourceImage.currentSrc || sourceImage.src;
    lightboxImage.alt = sourceImage.alt || "Image preview";
    lightbox.classList.add("is-open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.classList.remove("is-open");
    lightboxImage.removeAttribute("src");
    document.body.style.overflow = "";
  }

  document.addEventListener("click", (event) => {
    const image = event.target.closest(".visual-preview-cell img");
    if (!image) return;
    event.preventDefault();
    openLightbox(image);
  });

  document.addEventListener("click", (event) => {
    const removeButton = event.target.closest("[data-remove-visual-row]");
    if (removeButton) {
      event.preventDefault();
      const row = removeButton.closest("tr");
      if (row) row.remove();
      return;
    }

    const addButton = event.target.closest("[data-add-visual-row]");
    if (!addButton) return;
    event.preventDefault();
    const form = addButton.closest(".visual-edit-form");
    if (!form) return;
    const template = form.querySelector("template[data-visual-row-template]");
    const body = form.querySelector("[data-visual-rows]");
    if (!template || !body) return;
    const token = `new-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    const html = template.innerHTML.replaceAll("__TOKEN__", token);
    const wrapper = document.createElement("tbody");
    wrapper.innerHTML = html.trim();
    const row = wrapper.firstElementChild;
    if (row) {
      body.appendChild(row);
      const firstInput = row.querySelector("input[name='element-name']");
      if (firstInput) firstInput.focus();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
})();
