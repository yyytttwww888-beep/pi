import { fauxAssistantMessage, type ImageContent, type UserMessage } from "@earendil-works/pi-ai";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { loadPhoton } from "../../src/utils/photon.ts";
import { createHarness, getMessageText, type Harness } from "./harness.ts";

let screenshot: ImageContent;

beforeAll(async () => {
	const photon = await loadPhoton();
	if (!photon) throw new Error("Photon is required to exercise real image processing");
	const image = new photon.PhotonImage(new Uint8Array(1200 * 9000 * 4).fill(255), 1200, 9000);
	try {
		screenshot = { type: "image", mimeType: "image/png", data: Buffer.from(image.get_bytes()).toString("base64") };
	} finally {
		image.free();
	}
});

async function dimensions(image: ImageContent): Promise<number[]> {
	const photon = await loadPhoton();
	if (!photon) throw new Error("Photon is required to decode the provider image");
	const decoded = photon.PhotonImage.new_from_byteslice(Buffer.from(image.data, "base64"));
	try {
		return [decoded.get_width(), decoded.get_height()];
	} finally {
		decoded.free();
	}
}

function getImage(message: UserMessage): ImageContent {
	if (typeof message.content === "string") throw new Error("Expected image content");
	const image = message.content.find((part) => part.type === "image");
	if (!image) throw new Error("Expected an image in the user message");
	return image;
}

describe("AgentSession queued images", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	for (const entryPoint of ["prompt", "queue"] as const) {
		it.each(["steer", "followUp"] as const)(
			`${entryPoint}: normalizes %s images before history and requests`,
			async (mode) => {
				const started = Promise.withResolvers<void>();
				const release = Promise.withResolvers<void>();
				const harness = await createHarness();
				harnesses.push(harness);
				const requests: UserMessage[] = [];
				harness.setResponses([
					async () => {
						started.resolve();
						await release.promise;
						return fauxAssistantMessage("first response");
					},
					(context) => {
						requests.push(
							...context.messages.filter((message): message is UserMessage => message.role === "user"),
						);
						return fauxAssistantMessage("image received");
					},
				]);
				const run = harness.session.prompt("start");
				await started.promise;
				try {
					if (entryPoint === "prompt") {
						await harness.session.prompt("screenshot", { images: [screenshot], streamingBehavior: mode });
					} else {
						await harness.session[mode]("screenshot", [screenshot]);
					}
					expect(harness.session.pendingMessageCount).toBe(1);
				} finally {
					release.resolve();
					await run;
				}

				const received = requests.find((message) => getMessageText(message).startsWith("screenshot"));
				if (!received) throw new Error("Queued user message did not reach the provider");
				expect(await dimensions(getImage(received))).toEqual([267, 2000]);
				expect(getMessageText(received)).toContain("original 1200x9000, displayed at 267x2000");
				expect(harness.session.pendingMessageCount).toBe(0);
				const saved = harness.sessionManager
					.getBranch()
					.findLast((entry) => entry.type === "message" && entry.message.role === "user");
				expect(saved?.type === "message" && saved.message).toEqual(received);

				let nextImage: ImageContent | undefined;
				harness.appendResponses([
					(context) => {
						const previous = context.messages.find(
							(message): message is UserMessage =>
								message.role === "user" && getMessageText(message).startsWith("screenshot"),
						);
						if (previous) nextImage = getImage(previous);
						return fauxAssistantMessage("continued");
					},
				]);
				await harness.session.prompt("continue with text only");
				expect(nextImage).toEqual(getImage(received));
			},
		);
	}

	it("preserves queue order and uses the model selected before delivery", async () => {
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const harness = await createHarness({ models: [{ id: "initial" }, { id: "strict" }] });
		harnesses.push(harness);
		const strict = harness.getModel("strict");
		if (!strict) throw new Error("Expected strict model");
		strict.inputLimits = { images: { resize: { maxWidth: 900, maxHeight: 900 } } };
		harness.session.setSteeringMode("all");
		let received: UserMessage[] = [];
		harness.setResponses([
			async () => {
				started.resolve();
				await release.promise;
				return fauxAssistantMessage("first response");
			},
			(context) => {
				received = context.messages.filter((message): message is UserMessage => message.role === "user");
				return fauxAssistantMessage("done");
			},
		]);
		const run = harness.session.prompt("start");
		await started.promise;
		try {
			await Promise.all([
				harness.session.steer("screenshot", [screenshot]),
				harness.session.steer("second queued message"),
			]);
			await harness.session.setModel(strict);
		} finally {
			release.resolve();
			await run;
		}
		expect(received.map((message) => getMessageText(message).split("\n")[0])).toEqual([
			"start",
			"screenshot",
			"second queued message",
		]);
		expect(await dimensions(getImage(received[1]))).toEqual([120, 900]);
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	it("honors disabled image resizing for queued messages", async () => {
		const harness = await createHarness({ settings: { images: { autoResize: false } } });
		harnesses.push(harness);
		let received: ImageContent | undefined;
		harness.setResponses([
			(context) => {
				const queued = context.messages.find(
					(message): message is UserMessage => message.role === "user" && getMessageText(message) === "screenshot",
				);
				if (queued) received = getImage(queued);
				return fauxAssistantMessage("done");
			},
		]);
		await harness.session.steer("screenshot", [screenshot]);
		await harness.session.prompt("start");
		expect(received).toEqual(screenshot);
	});

	it("omits an undecodable queued image and keeps an explanation", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		let received: UserMessage | undefined;
		harness.setResponses([
			(context) => {
				received = context.messages.find(
					(message): message is UserMessage =>
						message.role === "user" && getMessageText(message).startsWith("screenshot"),
				);
				return fauxAssistantMessage("done");
			},
		]);
		await harness.session.steer("screenshot", [
			{ type: "image", mimeType: "image/png", data: Buffer.from("not an image").toString("base64") },
		]);
		await harness.session.prompt("start");
		if (!received || typeof received.content === "string") throw new Error("Expected queued user content");
		expect(received.content.some((part) => part.type === "image")).toBe(false);
		expect(getMessageText(received)).toContain("[Image omitted:");
	});
});
