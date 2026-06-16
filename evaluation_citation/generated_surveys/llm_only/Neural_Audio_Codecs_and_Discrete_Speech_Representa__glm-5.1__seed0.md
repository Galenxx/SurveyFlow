# 1 Abstract
The rapid advancement of generative models for audio has necessitated the development of efficient and high-fidelity intermediate representations. This survey provides a comprehensive overview of neural audio codecs and discrete speech representations, exploring their architectures, tokenization strategies, and generative applications. Traditionally, audio generation relied on continuous spectrogram features, which posed significant challenges for autoregressive modeling and long-context scaling. The advent of neural audio codecs, such as SoundStream and EnCodec, has fundamentally revolutionized this paradigm by enabling the compression of raw waveforms into discrete acoustic tokens using residual vector quantization (RVQ). We systematically examine the architectural foundations of these codecs, including convolutional encoder-decoder structures, multi-scale discriminators, and hierarchical quantization modules. Furthermore, we delve into advanced tokenization methods that explicitly disentangle semantic and acoustic features, enhancing the efficiency and interpretability of discrete representations for language modeling. The survey highlights how these tokenized representations serve as the foundational backbone for modern audio language models, facilitating unprecedented breakthroughs in zero-shot text-to-speech, voice cloning, music generation, and general audio synthesis. By synthesizing findings from recent seminal works, we illustrate the paradigm shift from continuous to discrete audio modeling and its profound implications for scalable generative AI. Finally, we outline current limitations and propose compelling future research directions, emphasizing the critical need for more robust, low-latency, and generalized codec frameworks capable of unifying diverse audio generation tasks across varying bandwidths.

# 2 Introduction
Audio generation has historically been a challenging domain within machine learning due to the high-dimensional, continuous, and complex nature of raw waveforms. Traditional approaches predominantly relied on generating continuous intermediate representations, such as mel-spectrograms, which were subsequently converted into waveforms using vocoders. While effective for limited tasks

# References
Reference: Unknown Author. (2024). Placeholder Reference 1. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 2. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 3. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 4. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 5. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 6. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 7. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 8. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 9. arXiv:0000.00000v1.
Reference: Unknown Author. (2024). Placeholder Reference 10. arXiv:0000.00000v1.