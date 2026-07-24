function refreshIcons(root = document) {
  if (!window.lucide) return;
  const placeholders = [...root.querySelectorAll("i[data-lucide]")];
  if (root.matches?.("i[data-lucide]")) placeholders.unshift(root);
  if (!placeholders.length) return;
  placeholders.forEach((icon) => {
    icon.dataset.lucidePending = icon.dataset.lucide;
    icon.removeAttribute("data-lucide");
  });
  window.lucide.createIcons({ nameAttr: "data-lucide-pending" });
  document.querySelectorAll("[data-lucide-pending]").forEach((icon) => {
    icon.removeAttribute("data-lucide-pending");
  });
}

function syncShotLocks(root = document) {
  const runtimes = [...root.querySelectorAll("[data-shot-runtime]")];
  if (root.matches?.("[data-shot-runtime]")) runtimes.unshift(root);
  runtimes.forEach((runtime) => {
    const section = runtime.closest(".shot-block");
    const form = section?.querySelector("[data-shot-form]");
    if (!form) return;
    const editable = ["draft", "queued", "stale", "failed", "interrupted"].includes(runtime.dataset.shotState);
    form.querySelectorAll("textarea, input[type='checkbox'], input[type='number'], select, button[type='submit']").forEach((control) => {
      control.disabled = !editable;
    });
    syncGenerationMode(form, editable);
  });
}

function syncGenerationMode(form, editable = true) {
  const cut = form.querySelector("[data-cut-toggle]");
  const mode = form.querySelector("[data-generation-mode]");
  if (!cut || !mode) return;
  const unavailable = cut.checked || form.dataset.hasPrevious !== "true";
  if (unavailable) mode.value = "default";
  mode.disabled = !editable || unavailable;
  mode.closest(".generation-mode-field")?.classList.toggle("is-disabled", mode.disabled);
  const value = cut.closest(".switch-control")?.querySelector("[data-switch-value]");
  if (value) value.textContent = cut.checked ? "On" : "Off";
}

function syncSaveAll() {
  const button = document.querySelector("[data-save-all]");
  if (button) button.disabled = !document.querySelector("[data-shot-form].is-dirty");
}

const pollScrollPositions = new WeakMap();

function isPollingTarget(target) {
  return target?.matches?.("[data-shot-runtime], #project-summary");
}

document.addEventListener("DOMContentLoaded", () => {
  refreshIcons();
  syncShotLocks();
  document.querySelectorAll("[data-shot-form]").forEach((form) => {
    syncGenerationMode(form, !form.querySelector("textarea")?.disabled);
    const markDirty = () => {
      form.classList.add("is-dirty");
      syncSaveAll();
    };
    form.addEventListener("input", markDirty);
    form.addEventListener("change", markDirty);
    form.querySelector("[data-cut-toggle]")?.addEventListener("change", () => {
      syncGenerationMode(form);
    });
  });
  document.addEventListener("submit", (event) => {
    if (!event.submitter?.matches("[data-run-action]")) return;
    if (!document.querySelector("[data-shot-form].is-dirty")) return;
    event.preventDefault();
    window.alert("Save edited shots before starting generation.");
  });

  document.addEventListener("click", async (event) => {
    const saveAll = event.target.closest("[data-save-all]");
    if (saveAll) {
      const forms = [...document.querySelectorAll("[data-shot-form].is-dirty")];
      saveAll.disabled = true;
      try {
        for (const form of forms) {
          const payload = {
            video_prompt: form.elements.video_prompt.value,
            is_cut: form.elements.is_cut.checked,
            generation_mode: form.elements.generation_mode.value,
            duration_seconds: Number(form.elements.duration_seconds.value),
            expected_row_version: Number(form.elements.row_version.value),
          };
          if (form.elements.memory_sink) payload.memory_sink = form.elements.memory_sink.checked;
          if (form.elements.memory_retrieve) payload.memory_retrieve = form.elements.memory_retrieve.checked;
          if (form.elements.memory_recent) payload.memory_recent = form.elements.memory_recent.checked;
          const response = await fetch(`/api${new URL(form.action).pathname}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(body.detail || `Save failed (${response.status})`);
          }
        }
        window.location.reload();
      } catch (error) {
        saveAll.disabled = false;
        window.alert(error.message || "Unable to save edited shots.");
      }
      return;
    }

    const button = event.target.closest("[data-keyframe-retry] button, [data-postprocess-retry] button");
    if (!button) return;
    const form = button.form;
    button.disabled = true;
    try {
      const response = await fetch(form.dataset.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: form.dataset.mode || "keyframes",
          shot_id: form.dataset.shotId,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Request failed (${response.status})`);
      }
      window.location.reload();
    } catch (error) {
      button.disabled = false;
      window.alert(error.message || "Unable to retry post-processing.");
    }
  });

  document.addEventListener("htmx:beforeSwap", (event) => {
    if (!isPollingTarget(event.detail.target) || !event.detail.xhr) return;
    pollScrollPositions.set(event.detail.xhr, { x: window.scrollX, y: window.scrollY });
  });

  document.addEventListener("htmx:afterSwap", (event) => {
    refreshIcons(event.detail.target);
    syncShotLocks(event.detail.target);
    const position = event.detail.xhr && pollScrollPositions.get(event.detail.xhr);
    if (!position) return;
    pollScrollPositions.delete(event.detail.xhr);
    window.requestAnimationFrame(() => {
      window.scrollTo({ left: position.x, top: position.y, behavior: "auto" });
    });
  });
});
